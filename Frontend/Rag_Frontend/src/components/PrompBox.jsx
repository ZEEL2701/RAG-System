import React from "react";
import profile from "../assets/profile.jpg";

const PrompBox = ({ setHasMessages, large }) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (setHasMessages) setHasMessages(true); // Move PrompBox to input area after first message
  };

  return (
    <form
      onSubmit={handleSubmit}
      className={`flex items-center w-full relative bg-[#F5F0E5] dark:bg-gray-700 border border-gray-300 
        dark:border-gray-600 rounded-lg px-3 py-1 transition-all 
        ${large ? "h-20" : "h-auto"}`} // Larger height when no chat
    >
      {/* Avatar Image */}
      <img src={profile} alt="User Avatar" className="w-8 h-8 rounded-full mr-3" />

      {/* Input Field */}
      <input
        type="text"
        id="default-input"
        className="w-full h-10 bg-transparent text-gray-900 dark:text-white 
             placeholder-[#A1824A] dark:placeholder-[#A1824A] 
             focus:outline-none focus:ring-0 focus:border-transparent border-none px-3 pr-20"
        placeholder="Message SoIBI"
      />

      {/* "+" Button */}
      <button
        type="button"
        className="absolute right-20 top-1/2 transform -translate-y-1/2 hover:opacity-80"
      >
        <svg className="w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 14 14">
          <path stroke="#A1824A" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M7 1v12M1 7h12" />
        </svg>
      </button>

      {/* Send Button */}
      <button
        type="submit"
        className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-green-400
                  text-white rounded-lg w-16 h-8 flex items-center justify-center 
                  shadow-md hover:opacity-80 transition px-2 text-sm"
      >
        Send
      </button>
    </form>
  );
};

export default PrompBox;
