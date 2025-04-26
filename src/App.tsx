import React, { useEffect, useState } from 'react';
import Scene from './components/Scene'

function App() {
  const [buttonClicks, setButtonClicks] = useState(0);

  const handleButtonClick = () => {
    setButtonClicks(prev => prev + 1);
    console.log('Button clicked!');
  };

  return (
    <div className="app-container">
      <div className='scene-container'>
        <Scene />
      </div>
      <div className="controls-panel">
        <button
          className="control-button"
          onClick={handleButtonClick}
        >
          Click Me ({buttonClicks})
        </button>
      </div>

    </div>
  )
}

export default App

