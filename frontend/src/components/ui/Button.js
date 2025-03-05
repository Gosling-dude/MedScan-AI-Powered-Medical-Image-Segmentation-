export function Button({ onClick, children }) {
    return (
      <button
        onClick={onClick}
        className="w-full bg-blue-500 text-white py-2 rounded-lg hover:bg-blue-600 transition"
      >
        {children}
      </button>
    );
  }
  