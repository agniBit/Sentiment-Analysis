import React from "react";
import './App.css';
import axios from 'axios';

export default function App() {
  const [text, setText] = React.useState("");
  const [output,setOutput] = React.useState("");

  function setOut(t){
    setOutput(t);
  }

  const handleSubmit = async(event) => {
    const j_data = {
      text
    }
    axios.post(
      "http://127.0.0.1:6543/get_data", 
       JSON.stringify(j_data)
      ).then(resp => {
        setOut(resp.data.output);
    });
    event.preventDefault();
  }

  return (
    <div className="App">
      <div className='form-container'>
        <form onSubmit={handleSubmit}>
          <h1>Sentiment Analysis</h1>
            <textarea
              name="text"
              type="text"
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder='text'
              required />
          <div className={output}>
            {output}
          </div>
          <button>Check Sentiment</button>
        </form>
      </div>
    </div>
  );
}