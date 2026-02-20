const ChatHeader = () => {
  return (
    <header 
      className="h-8 flex items-center px-4 border-b border-[#00f0ff]/20"
      style={{ background: 'rgba(5, 5, 8, 0.95)' }}
    >
      <div className="w-1.5 h-1.5 bg-[#00f0ff] animate-pulse" />
    </header>
  );
};

export default ChatHeader;
