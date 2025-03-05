export function Input({ type, value, onChange, accept }) {
    return (
      <input
        type={type}
        value={value}
        onChange={onChange}
        accept={accept}
        className="w-full p-2 border rounded-lg"
      />
    );
  }
  