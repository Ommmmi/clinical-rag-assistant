import { useState, useRef, useEffect } from 'react';

type Message = {
  text: string;
  sender: 'user' | 'bot';
  timestamp: string;
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      text: "Hello! I'm your MediCare AI Assistant. How can I help you today?",
      sender: 'bot',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(import.meta.env.VITE_API_URL || 'http://localhost:8080/get', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ msg: inputValue }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch response');
      }

      const data = await response.json();
      
      const botMessage: Message = {
        text: data.answer,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        text: "Sorry, I'm having trouble connecting to the server. Please make sure the backend is running.",
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="size-full bg-gradient-to-br from-gray-900 via-blue-950 to-gray-900 flex items-center justify-center p-4 relative overflow-hidden h-screen">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-20 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="w-full max-w-4xl h-[90vh] bg-gradient-to-br from-gray-800/90 to-gray-900/90 backdrop-blur-xl rounded-3xl shadow-2xl flex flex-col overflow-hidden border border-gray-700/50 relative z-10">
        
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 via-cyan-600 to-blue-600 bg-[length:200%_100%] animate-[gradient_3s_ease_infinite] text-white px-6 py-5 flex items-center gap-4 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-[shimmer_2s_ease-in-out_infinite]"></div>
          <div className="w-12 h-12 bg-white/95 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg transform hover:scale-110 transition-transform duration-300 relative z-10">
            <svg className="w-7 h-7 text-blue-600 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
            </svg>
          </div>
          <div className="flex-1 relative z-10">
            <h1 className="text-xl font-semibold tracking-wide">MediCare AI Assistant</h1>
            <p className="text-blue-100 text-sm flex items-center gap-2">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-ping absolute"></span>
              <span className="w-2 h-2 bg-green-400 rounded-full relative"></span>
              Online • Powered by AI
            </p>
          </div>
        </div>

        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-gradient-to-b from-gray-900/50 to-gray-800/50">
          
          {messages.map((msg, index) => (
            <div 
              key={index} 
              className={`flex gap-3 items-start ${msg.sender === 'user' ? 'justify-end' : ''} animate-[fadeInUp_0.3s_ease-out_forwards]`}
            >
              {msg.sender === 'bot' && (
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg shadow-blue-500/50 hover:scale-110 transition-transform duration-300">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
              )}
              
              <div className={`flex-1 flex flex-col ${msg.sender === 'user' ? 'items-end' : ''}`}>
                <div className={`
                  ${msg.sender === 'user' 
                    ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-tr-none' 
                    : 'bg-gradient-to-br from-gray-700/50 to-gray-800/50 text-gray-100 rounded-tl-none border border-gray-600/30'} 
                  rounded-2xl px-5 py-4 shadow-xl max-w-md hover:scale-[1.01] transition-transform duration-300
                `}>
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                </div>
                <span className={`text-xs text-gray-500 mt-1 ${msg.sender === 'user' ? 'mr-2' : 'ml-2'}`}>
                  {msg.timestamp}
                </span>
              </div>

              {msg.sender === 'user' && (
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg shadow-blue-500/50 hover:scale-110 transition-transform duration-300 border-2 border-blue-400/30">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
              )}
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex gap-3 items-start animate-[fadeInUp_0.3s_ease-out_forwards]">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg shadow-blue-500/50 animate-pulse">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div className="bg-gradient-to-br from-gray-700/50 to-gray-800/50 backdrop-blur-sm rounded-2xl rounded-tl-none px-5 py-3 shadow-xl border border-blue-500/30">
                <div className="flex gap-1.5">
                  <span className="w-2.5 h-2.5 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full animate-bounce shadow-lg shadow-blue-500/50" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-2.5 h-2.5 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full animate-bounce shadow-lg shadow-cyan-500/50" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-2.5 h-2.5 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full animate-bounce shadow-lg shadow-blue-500/50" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-gradient-to-br from-gray-800/95 to-gray-900/95 backdrop-blur-xl border-t border-gray-700/50 p-4">
          <form onSubmit={handleSendMessage} className="flex gap-3 items-end">
            
            <div className="flex-1 bg-gradient-to-br from-gray-700/50 to-gray-800/50 backdrop-blur-sm rounded-2xl px-4 py-3 flex items-center gap-3 border border-gray-600/50 focus-within:border-blue-500/50 transition-all duration-300 shadow-lg">
              <input 
                type="text" 
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Type your message here..." 
                className="flex-1 bg-transparent outline-none text-gray-100 placeholder-gray-400"
                disabled={isLoading}
              />
            </div>
            <button 
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className="w-11 h-11 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-300 shadow-lg shadow-blue-500/50 hover:shadow-blue-500/70 hover:scale-110 border border-blue-400/30 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </form>
          <p className="text-xs text-gray-500 text-center mt-3 flex items-center justify-center gap-2">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            End-to-end encrypted • This chatbot provides general health information and is not a substitute for professional medical advice.
          </p>
        </div>

      </div>
    </div>
  );
}
