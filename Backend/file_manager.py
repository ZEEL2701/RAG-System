import os
import json
import shutil
import logging
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("enhanced_rag")

class FileManager:
    def __init__(self, config, s3_manager):
        self.config = config
        self.s3_manager = s3_manager
        self.local_storage_path = os.path.join(os.getcwd(), "uploaded_files")
        os.makedirs(self.local_storage_path, exist_ok=True)
        self.config.setup_file_registry_db()

    def register_file(self, file_path: str, session_id: str, object_key: Optional[str] = None, metadata: Optional[Dict] = None) -> Optional[int]:
        """Register a file in the database."""
        file_name = os.path.basename(file_path)
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM file_registry WHERE file_name = %s AND session_id = %s",
                (file_name, session_id)
            )
            if cursor.fetchone():
                logger.info(f"File already registered: {file_name}")
                return None

            file_type = os.path.splitext(file_name)[1].lower()[1:]  # Remove the dot

            cursor.execute(
                """
                INSERT INTO file_registry
                (file_name, file_path, file_type, object_key, session_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    file_name,
                    file_path,
                    file_type,
                    object_key,
                    session_id,
                    json.dumps(metadata or {})
                )
            )
            file_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Registered file with ID {file_id}: {file_name}")
            return file_id
        except Exception as e:
            logger.error(f"Failed to register file: {str(e)}")
            return None
        finally:
            cursor.close()
            conn.close()

    def store_file(self, source_path: str, session_id: str, metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Store a file locally and optionally upload to S3, then register it."""
        file_name = os.path.basename(source_path)
        local_path = os.path.join(self.local_storage_path, f"{session_id}_{file_name}")

        try:
            shutil.copy2(source_path, local_path)
            logger.info(f"Stored local copy at {local_path}")

            object_key = None
            if self.config.s3_enabled:
                object_key = self.s3_manager.upload_file(source_path)

            file_id = self.register_file(local_path, session_id, object_key, metadata)

            result = {
                "file_id": file_id,
                "file_name": file_name,
                "local_path": local_path,
                "object_key": object_key,
                "session_id": session_id
            }

            if object_key:
                result["download_url"] = self.s3_manager.generate_presigned_url(object_key)

            return result
        except Exception as e:
            logger.error(f"Failed to store file: {str(e)}")
            return None

    def list_files(self, session_id: Optional[str] = None) -> List[Dict]:
        """List files from the registry, optionally filtered by session_id."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            if session_id:
                cursor.execute(
                    """
                    SELECT id AS file_id, file_name, file_path, file_type, object_key, upload_date, metadata
                    FROM file_registry
                    WHERE session_id = %s
                    ORDER BY upload_date DESC;
                    """,
                    (session_id,)
                )
            else:
                cursor.execute(
                    """
                    SELECT id AS file_id, file_name, file_path, file_type, object_key, upload_date, metadata
                    FROM file_registry
                    ORDER BY upload_date DESC;
                    """
                )

            files = []
            for row in cursor.fetchall():
                file_info = dict(row)
                file_info["upload_date"] = file_info["upload_date"].strftime("%Y-%m-%d %H:%M:%S")
                if file_info.get("object_key"):
                    try:
                        file_info["download_url"] = self.s3_manager.generate_presigned_url(file_info["object_key"])
                    except Exception as e:
                        logger.error(f"Failed to generate presigned URL for {file_info['file_name']}: {str(e)}")
                files.append(file_info)
            return files
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_file(self, file_id: int) -> Optional[Dict]:
        """Get details for a specific file by ID."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """
                SELECT id AS file_id, file_name, file_path, file_type, object_key, upload_date, metadata
                FROM file_registry
                WHERE id = %s;
                """,
                (file_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            file_info = dict(row)
            file_info["upload_date"] = file_info["upload_date"].strftime("%Y-%m-%d %H:%M:%S")
            if file_info.get("object_key"):
                try:
                    file_info["download_url"] = self.s3_manager.generate_presigned_url(file_info["object_key"])
                except Exception as e:
                    logger.error(f"Failed to generate presigned URL for {file_info['file_name']}: {str(e)}")
            return file_info
        except Exception as e:
            logger.error(f"Failed to get file info: {str(e)}")
            return None
        finally:
            cursor.close()
            conn.close()

    def delete_file(self, file_id: int) -> bool:
        """Delete a file from storage and registry."""
        file_info = self.get_file(file_id)
        if not file_info:
            return False

        # Delete from S3 if enabled
        if self.config.s3_enabled and file_info.get("object_key"):
            self.s3_manager.delete_file(file_info["object_key"])

        # Delete local file
        if os.path.exists(file_info["file_path"]):
            try:
                os.remove(file_info["file_path"])
            except Exception as e:
                logger.error(f"Failed to delete local file: {str(e)}")

        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM file_registry WHERE id = %s;", (file_id,))
            conn.commit()
            logger.info(f"Deleted file with ID {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file from registry: {str(e)}")
            return False
        finally:
            cursor.close()
            conn.close()
