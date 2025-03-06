import React, { useState } from "react";
import PrompBox from "./components/PrompBox";

const App = () => {
  const [hasMessages, setHasMessages] = useState(false); // Tracks if chat has messages

  return (
    <div className="w-screen h-screen grid grid-cols-5 grid-rows-6 gap-0">
      {/* Header (Untouched) */}
      <div className="col-span-5 row-start-1 bg-gray-800 text-white flex items-center justify-center">
        Header Content
      </div>

      {/* Sidebar */}
      <div className="row-span-5 col-start-1 row-start-2 bg-green-500">3</div>

      {/* Main Chat Area */}
      <div className="col-span-4 row-span-4 col-start-2 row-start-2 bg-pink-900 flex justify-center items-center">
        {!hasMessages && ( // If no messages, show large PrompBox in the center
          <div className="w-full max-w-[700px] px-4 md:px-8 lg:px-16">
            <PrompBox setHasMessages={setHasMessages} large={true} />
          </div>
        )}
      </div>

      {/* Input box (Shifts here after first message) */}
      {hasMessages && (
        <div className="col-span-4 col-start-2 row-start-6 bg-blue-400 flex justify-center items-center">
          <div className="w-full max-w-[700px] px-4 md:px-8 lg:px-16">
            <PrompBox />
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
