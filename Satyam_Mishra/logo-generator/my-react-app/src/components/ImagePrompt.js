// src/ImagePrompt.js
import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const ImagePrompt = () => {
    const [prompt, setPrompt] = useState('');
    const [negprompt, setNegPrompt] = useState('');
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleInputChange = (e) => {
        setPrompt(e.target.value);
    };
    const handleNegInputChange = (e) => {
        setNegPrompt(e.target.value);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            const response = await axios.post('https://8000-01jb9r7naszt1w8rvy1hcvw1tb.cloudspaces.litng.ai/api/logo/generate', {
                prompt,
                negprompt
            });
            setImages(response.data.images);
        } catch (err) {
            setError('Failed to fetch images. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleImageClick = (image) => {
        // Encode image data to be passed in URL
        localStorage.setItem('editImage', image);  // Store base64 in local storage
        navigate(`/edit_logo`);
    };

    const handleSaveImage = (image, index) => {
        const link = document.createElement('a');
        link.href = `data:image/png;base64,${image}`;
        link.download = `generated_image_${index + 1}.png`;
        link.click();
    };

    return (
        <div className="max-w-lg mx-auto p-6 bg-white rounded-lg shadow-md">
            <h1 className="text-2xl font-bold text-center mb-4">AI Logo Generator</h1>
            <form onSubmit={handleSubmit} className="flex flex-col">
                <input
                    type="text"
                    value={prompt}
                    onChange={handleInputChange}
                    placeholder="Enter your prompt"
                    required
                    className="p-2 border border-gray-300 rounded mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <input
                    type="text"
                    value={negprompt}
                    onChange={handleNegInputChange}
                    placeholder="Enter your negative prompt"
                    required
                    className="p-2 border border-gray-300 rounded mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                    type="submit"
                    disabled={loading}
                    className={`p-2 text-white rounded ${loading ? 'bg-gray-400' : 'bg-blue-500 hover:bg-blue-600'} transition duration-200`}
                >
                    {loading ? 'Sending...' : 'Send Prompt'}
                </button>
            </form>
            {error && <p className="text-red-500 mt-4">{error}</p>}
            <div className="mt-6">
                {images.length > 0 && (
                    <h2 className="text-xl font-semibold mb-2">Generated Images:</h2>
                )}
                <div className="grid grid-cols-2 gap-4">
                    {images.map((image, index) => (
                        <div key={index} className="relative">
                            <img
                                src={`data:image/png;base64,${image}`}
                                alt={`Generated ${index}`}
                                className="w-full h-auto rounded shadow cursor-pointer mb-2"
                                onClick={() => handleImageClick(image)}
                            />
                            <button
                                onClick={() => handleSaveImage(image, index)}
                                className="absolute bottom-2 right-2 bg-blue-500 text-white px-2 py-1 rounded text-xs"
                            >
                                Save
                            </button>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default ImagePrompt;
