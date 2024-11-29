// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import ImagePrompt from './components/ImagePrompt';
import EditLogo from './components/EditLogo';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ImagePrompt />} />
        <Route path="/edit_logo" element={<EditLogo />} />
      </Routes>
    </Router>
  );
};

export default App;
