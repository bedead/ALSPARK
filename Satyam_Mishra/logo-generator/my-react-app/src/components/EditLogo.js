import React, { useEffect, useRef, useState } from 'react';
import { Stage, Layer, Image as KonvaImage, Text, Rect } from 'react-konva';
import { Button, Slider, TextField, IconButton, MenuItem, Select, Box } from '@mui/material';
import { Undo, Redo, RotateLeft, RotateRight, Save, Delete, Add, FilterNone } from '@mui/icons-material';
import useImage from 'use-image';
import Konva from 'konva';

const EditLogo = () => {
    const [imageUrl, setImageUrl] = useState(null);
    const [image] = useImage(imageUrl);  // Use the image URL with useImage hook

    useEffect(() => {
        // Get the byte string from localStorage
        const byteString = localStorage.getItem('editImage');

        // Convert byte string to a base64 Data URL
        const base64DataUrl = `data:image/png;base64,${byteString}`;
        setImageUrl(base64DataUrl);  // Set the URL for useImage to load
    }, []);

    const [texts, setTexts] = useState([]);
    const [textInput, setTextInput] = useState('');
    const [selectedTextId, setSelectedTextId] = useState(null);
    const stageRef = useRef(null);

    const [canvasDimensions, setCanvasDimensions] = useState({
        width: window.innerWidth * 0.7,
        height: window.innerHeight * 0.7,
    });

    const [imageProps, setImageProps] = useState({ x: 0, y: 0, scaleX: 1, scaleY: 1 });

    useEffect(() => {
        const byteString = localStorage.getItem('editImage');
        const base64DataUrl = `data:image/png;base64,${byteString}`;
        setImageUrl(base64DataUrl);
    }, []);

    useEffect(() => {
        const handleResize = () => {
            setCanvasDimensions({
                width: window.innerWidth * 0.6,
                height: window.innerHeight * 0.6,
            });
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    useEffect(() => {
        if (image) {
            // Calculate aspect ratio scaling to fit within canvas dimensions without cropping
            const imageAspectRatio = image.width / image.height;
            const canvasAspectRatio = canvasDimensions.width / canvasDimensions.height;

            let newScaleX, newScaleY;
            if (canvasAspectRatio > imageAspectRatio) {
                newScaleX = newScaleY = canvasDimensions.height / image.height;
            } else {
                newScaleX = newScaleY = canvasDimensions.width / image.width;
            }

            setImageProps({
                x: (canvasDimensions.width - image.width * newScaleX) / 2,
                y: (canvasDimensions.height - image.height * newScaleY) / 2,
                scaleX: newScaleX,
                scaleY: newScaleY,
            });
        }
    }, [image, canvasDimensions]);

    const handleInputChange = (e) => {
        setTextInput(e.target.value);
    };

    const handleTextUpdate = (id, attrs) => {
        setTexts(texts.map((text) => (text.id === id ? { ...text, ...attrs } : text)));
    };

    const addText = () => {
        const newText = {
            id: texts.length + 1,
            text: textInput,
            x: 50,
            y: 50,
            fontSize: 24,
            rotation: 0,
            draggable: true,
        };
        const updatedTexts = [...texts, newText];
        setTexts(updatedTexts);
        setTextInput('');
    };

    const downloadImage = (format) => {
        const uri = stageRef.current.toDataURL({ mimeType: format });
        const link = document.createElement('a');
        link.href = uri;
        link.download = `edited_image.${format.split('/')[1]}`;
        link.click();
    };

    const deleteText = (id) => {
        setTexts(texts.filter((text) => text.id !== id));
        setSelectedTextId(null);
    };

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 2 }}>
            <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                <TextField
                    type="text"
                    placeholder='Sample Text'
                    value={textInput}
                    onChange={handleInputChange}
                    size="small"
                    sx={{ minWidth: '100px' }}
                />
                <Button startIcon={<Add />} onClick={addText}>Add Text</Button>
                <Button startIcon={<Save />} onClick={() => downloadImage('image/png')}>Save PNG</Button>
                <Button startIcon={<Save />} onClick={() => downloadImage('image/jpeg')}>Save JPG</Button>
            </Box>
            <br />

            <Stage ref={stageRef} width={canvasDimensions.width} height={canvasDimensions.height} scaleX={1} scaleY={1}>
                <Layer>
                    <KonvaImage
                        image={image}
                        x={imageProps.x}
                        y={imageProps.y}
                        scaleX={imageProps.scaleX}
                        scaleY={imageProps.scaleY}
                    />
                    {texts.map((text) => (
                        <React.Fragment key={text.id}>
                            {/* {selectedTextId === text.id && (
                                <Rect
                                    x={text.x - 6.5} // Offset to make the rectangle fit nicely around the text
                                    y={text.y - 6.5} // Offset to align with the text
                                    width={text.text.length * text.fontSize * 0.6} // Approximate width of text
                                    height={text.fontSize + 10} // Height of text plus padding
                                    stroke="black"
                                    onClick={() => setSelectedTextId(text.id)} // Keep the text selected on click
                                />
                            )} */}
                            <Text
                                {...text}
                                onClick={() => setSelectedTextId(text.id)}
                                onTransform={(e) => {
                                    const node = e.target;
                                    handleTextUpdate(text.id, {
                                        rotation: node.rotation(),
                                        scaleX: node.scaleX(),
                                        scaleY: node.scaleY()
                                    });
                                }}
                                draggable
                                onDragEnd={(e) => handleTextUpdate(text.id, { x: e.target.x(), y: e.target.y() })}
                            // onDblClick={() => deleteText(text.id)}
                            />
                        </React.Fragment>
                    ))}
                </Layer>
            </Stage>
            <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                <Slider
                    value={texts.find((text) => text.id === selectedTextId)?.fontSize}
                    onChange={(e, value) => handleTextUpdate(selectedTextId, { fontSize: value })}
                    min={10}
                    max={100}
                    aria-labelledby="font-size-slider"
                    sx={{ width: '200px' }}
                />
                <Select
                    value={texts.find((text) => text.id === selectedTextId)?.fontFamily || 'Arial'}
                    onChange={(e) => handleTextUpdate(selectedTextId, { fontFamily: e.target.value })}
                    displayEmpty
                    inputProps={{ 'aria-label': 'Without label' }}
                >
                    <MenuItem value="Arial">Arial</MenuItem>
                    <MenuItem value="Courier New">Courier New</MenuItem>
                    <MenuItem value="Times New Roman">Times New Roman</MenuItem>
                </Select>
                <IconButton color="primary" onClick={() => deleteText(selectedTextId)}><Delete /></IconButton>
            </Box>
        </Box>
    );
};

export default EditLogo;
