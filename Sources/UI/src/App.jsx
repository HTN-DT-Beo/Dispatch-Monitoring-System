import { useState } from 'react';
function App() {
  const [vi, setVi] = useState(null);
  const submit = async e => {
    e.preventDefault();
    const fd = new FormData(); fd.append('file', vi);
    const res = await fetch('http://localhost:8000/infer', {method:'POST', body:fd});
    const {result} = await res.json(); window.open(`/videos/${result}`, '_blank');
  };
  return (
    <div className="flex flex-col items-center mt-10">
      <input type="file" accept="video/mp4" onChange={e=>setVi(e.target.files[0])}/>
      <button onClick={submit} className="bg-blue-500 text-white px-4 py-2 rounded">Upload & Infer</button>
    </div>
  );
}
export default App;
