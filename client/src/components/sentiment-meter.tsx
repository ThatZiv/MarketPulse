import GaugeComponent from 'react-gauge-component';
import { useTheme } from './ui/theme-provider';
import { useState, useEffect } from 'react';

interface SentimentMeterProps {
    score: number
}

export const SentimentMeter = ({ score }: SentimentMeterProps) => {
    const { theme } = useTheme();
    const [tickLabelColor, setTickLabelColor] = useState('');
    const [pointerColor, setPointerColor] = useState('');

    useEffect(() => {
        console.log("SentimentMeter theme: ", theme);
        const labelColor = theme === 'dark' ? '#FFFFFF' : '#000000';
        const pointerColor = theme === 'dark'? 'rgba(255, 255, 255, 0.5)' : 'rgba(0, 0, 0, 0.5)';
        console.log("Tick label color set to: ", labelColor); 
        setTickLabelColor(labelColor);
        setPointerColor(pointerColor);
    }, [theme]);
    return (
        <GaugeComponent
        key={tickLabelColor}
            className='w-full'
            type="semicircle"
            marginInPercent={0.08}
            arc={{
                colorArray: ['#FF2121', '#00FF15'],
                cornerRadius: 5,
                width:0.3,
                padding: 0.005,
                subArcs: [
                    {
                        limit: 20,
                        showTick: true,
                        tooltip: {
                            text: 'Mostly negative'
                        }
                    },
                    {
                        limit: 40,
                        showTick: true,
                        tooltip: {
                            text: 'Negative'
                        }
                    },
                    {
                        limit: 60,
                        showTick: true,
                        tooltip: {
                            text: 'Neutral'
                        }
                    },
                    {
                        limit: 80,
                        showTick: true,
                        tooltip: {
                            text: 'Positive'
                        }
                    },
                    {
                        limit: 100,
                        showTick: true,
                        tooltip: {
                            text: 'Mostly positive'
                        }
                    },
                ]
            }}
            value={score}
            minValue={0}
            maxValue={100}
            pointer={{
                type: "needle",
                elastic: true,
                color: pointerColor,
                animationDuration: 4000
            }}
            labels={{
                valueLabel: {
                    style: {
                        fontWeight: 700,
                        fill: tickLabelColor,
                    },
                },
                tickLabels: {
                    type: 'outer',
                    defaultTickValueConfig: {
                        style: {
                            fontSize: "1rem",
                            fill: tickLabelColor,
                            textShadow: theme === 'dark' ? 
                            'black 1px 1px 0px, black 0px 0px 2.5em, black 0px 0px 0.2em': 
                            '0px 0px 5px rgba(0, 255, 255, 0.4), 0px 0px 10px rgba(0, 255, 255, 0.2)',
                        }
                    },
                    defaultTickLineConfig: {
                        color: tickLabelColor,
                    }
                },

            }}

        />
    )
}

