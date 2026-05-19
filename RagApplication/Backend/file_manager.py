import os
import json
import shutil
import hashlib
import logging
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("enhanced_rag")


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class FileManager:
    def __init__(self, config, s3_manager):
        self.config = config
        self.s3_manager = s3_manager
        self.local_storage_path = os.path.join(os.getcwd(), "uploaded_files")
        os.makedirs(self.local_storage_path, exist_ok=True)
        self.config.setup_file_registry_db()

    def register_file(
        self,
        file_path: str,
        session_id: str,
        registry_file_name: str,
        object_key: Optional[str] = None,
        metadata: Optional[Dict] = None,
        content_hash: Optional[str] = None,
    ) -> Optional[int]:
        """Register a file. registry_file_name is unique per session (e.g. sessionId_original.pdf); file_path is physical storage."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM file_registry WHERE file_name = %s AND session_id = %s",
                (registry_file_name, session_id),
            )
            if cursor.fetchone():
                logger.info(f"File already registered: {registry_file_name}")
                return None

            ext = os.path.splitext(registry_file_name)[1].lower()
            file_type = ext[1:] if ext else "bin"

            cursor.execute(
                """
                INSERT INTO file_registry
                (file_name, file_path, file_type, object_key, session_id, metadata, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    registry_file_name,
                    file_path,
                    file_type,
                    object_key,
                    session_id,
                    json.dumps(metadata or {}),
                    content_hash,
                ),
            )
            file_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Registered file with ID {file_id}: {registry_file_name}")
            return file_id
        except Exception as e:
            logger.error(f"Failed to register file: {str(e)}")
            return None
        finally:
            cursor.close()
            conn.close()

    def get_file_by_session_and_name(self, session_id: str, registry_file_name: str) -> Optional[Dict]:
        """Return registry row where file_name is the per-session logical name (sessionId_original.ext)."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """
                SELECT id AS file_id, file_name, file_path, file_type, object_key, upload_date, metadata, content_hash
                FROM file_registry
                WHERE session_id = %s AND file_name = %s
                ORDER BY id DESC
                LIMIT 1;
                """,
                (session_id, registry_file_name),
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
            logger.error(f"Failed to look up file by session/name: {str(e)}")
            return None
        finally:
            cursor.close()
            conn.close()

    def _find_reusable_file_by_content_hash(self, content_hash: str) -> Optional[Dict]:
        """Return a row with this content_hash whose file still exists on disk (any session)."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """
                SELECT id AS file_id, file_name, file_path, file_type, object_key, upload_date, metadata, content_hash
                FROM file_registry
                WHERE content_hash = %s
                ORDER BY id ASC;
                """,
                (content_hash,),
            )
            for row in cursor.fetchall():
                info = dict(row)
                if os.path.exists(info["file_path"]):
                    info["upload_date"] = info["upload_date"].strftime("%Y-%m-%d %H:%M:%S")
                    if info.get("object_key"):
                        try:
                            info["download_url"] = self.s3_manager.generate_presigned_url(info["object_key"])
                        except Exception as e:
                            logger.error(f"Failed to generate presigned URL: {str(e)}")
                    return info
            return None
        except Exception as e:
            logger.error(f"Failed to look up file by content hash: {str(e)}")
            return None
        finally:
            cursor.close()
            conn.close()

    def store_file(self, source_path: str, session_id: str, metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Store by content hash: one physical copy per unique bytes (all sessions); one registry row per session+name."""
        original_name = os.path.basename(source_path)
        registry_file_name = f"{session_id}_{original_name}"

        try:
            existing = self.get_file_by_session_and_name(session_id, registry_file_name)
            if existing:
                if not os.path.exists(existing["file_path"]):
                    logger.warning(
                        f"Registry entry for {original_name} points to missing path {existing['file_path']}; removing stale row."
                    )
                    self.delete_file(existing["file_id"])
                else:
                    logger.info(f"File already stored for this session, skipping copy and re-index: {original_name}")
                    return {
                        "file_id": existing["file_id"],
                        "file_name": original_name,
                        "local_path": existing["file_path"],
                        "object_key": existing.get("object_key"),
                        "session_id": session_id,
                        "download_url": existing.get("download_url"),
                        "duplicate_skipped": True,
                    }

            content_hash = _sha256_file(source_path)
            ext = os.path.splitext(original_name)[1].lower()
            if not ext:
                ext = ".bin"
            physical_path = os.path.join(self.local_storage_path, f"{content_hash}{ext}")

            used_shared_path = False
            reusable = self._find_reusable_file_by_content_hash(content_hash)
            if reusable and os.path.exists(reusable["file_path"]):
                used_shared_path = True
                physical_path = reusable["file_path"]
                object_key = reusable.get("object_key")
                if self.config.s3_enabled and not object_key:
                    object_key = self.s3_manager.upload_file(physical_path)
                logger.info(
                    f"Reusing existing stored bytes for {original_name} (same content across sessions); path={physical_path}"
                )
            else:
                object_key = None
                if not os.path.exists(physical_path):
                    shutil.copy2(source_path, physical_path)
                    logger.info(f"Stored content-addressed copy at {physical_path}")
                else:
                    logger.info(f"Using existing on-disk file at {physical_path}")
                if self.config.s3_enabled:
                    object_key = self.s3_manager.upload_file(physical_path)

            file_id = self.register_file(
                physical_path,
                session_id,
                registry_file_name,
                object_key,
                metadata,
                content_hash,
            )
            if file_id is None:
                logger.error(f"register_file returned no id for {original_name}; registry may be inconsistent.")
                if not used_shared_path and os.path.exists(physical_path):
                    try:
                        if self._count_registry_refs_for_path(physical_path) == 0:
                            os.remove(physical_path)
                    except OSError:
                        pass
                return None

            result = {
                "file_id": file_id,
                "file_name": original_name,
                "local_path": physical_path,
                "object_key": object_key,
                "session_id": session_id,
                "duplicate_skipped": False,
                "storage_reused": used_shared_path,
            }

            if object_key:
                result["download_url"] = self.s3_manager.generate_presigned_url(object_key)

            return result
        except Exception as e:
            logger.error(f"Failed to store file: {str(e)}")
            return None

    def _count_registry_refs_for_path(self, file_path: str) -> int:
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM file_registry WHERE file_path = %s;",
                (file_path,),
            )
            return int(cursor.fetchone()[0])
        except Exception as e:
            logger.error(f"Failed to count registry refs: {str(e)}")
            return 0
        finally:
            cursor.close()
            conn.close()

    def list_files(self, session_id: Optional[str] = None) -> List[Dict]:
        """List files from the registry, optionally filtered by session_id."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            if session_id:
                cursor.execute(
                    """
                    SELECT id AS file_id, file_name, file_path, file_type, object_key, upload_date, metadata, content_hash
                    FROM file_registry
                    WHERE session_id = %s
                    ORDER BY upload_date DESC;
                    """,
                    (session_id,)
                )
            else:
                cursor.execute(
                    """
                    SELECT id AS file_id, file_name, file_path, file_type, object_key, upload_date, metadata, content_hash
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
                SELECT id AS file_id, file_name, file_path, file_type, object_key, upload_date, metadata, content_hash
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
        """Remove registry row; delete S3 and local file only when no other row references the same path."""
        file_info = self.get_file(file_id)
        if not file_info:
            return False

        storage_path = file_info["file_path"]
        object_key = file_info.get("object_key")

        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM file_registry WHERE file_path = %s;",
                (storage_path,),
            )
            ref_count = int(cursor.fetchone()[0])
            cursor.execute("DELETE FROM file_registry WHERE id = %s;", (file_id,))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to delete file from registry: {str(e)}")
            return False
        finally:
            cursor.close()
            conn.close()

        if ref_count <= 1:
            if self.config.s3_enabled and object_key:
                try:
                    self.s3_manager.delete_file(object_key)
                except Exception as e:
                    logger.error(f"Failed to delete S3 object: {str(e)}")
            if os.path.exists(storage_path):
                try:
                    os.remove(storage_path)
                except OSError as e:
                    logger.error(f"Failed to delete local file: {str(e)}")
        else:
            logger.info(
                f"Kept shared storage file (still referenced {ref_count - 1} time(s)): {storage_path}"
            )

        logger.info(f"Deleted registry row id={file_id}")
        return True
