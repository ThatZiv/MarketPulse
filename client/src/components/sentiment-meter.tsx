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
        const labelColor = theme === 'dark' ? '#FFFFFF' : '#000000';
        const pointerColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.5)' : 'rgba(0, 0, 0, 0.5)';
        setTickLabelColor(labelColor);
        setPointerColor(pointerColor);
    }, [theme]);
    return (
        <GaugeComponent
            key={tickLabelColor}
            className='w-full'
            type="semicircle"
            marginInPercent={0.07}
            arc={{
                colorArray: ['#FF2121', '#00FF15'],
                cornerRadius: 1,
                width: 0.3,
                padding: 0.005,
                subArcs: [
                    {
                        limit: 15,
                        showTick: true,
                        tooltip: {
                            text: 'Mostly Negative'
                        }
                    },
                    {
                        limit: 30,
                        showTick: true,
                        tooltip: {
                            text: 'Negative'
                        }
                    },
                    {
                        limit: 45,
                        showTick: true,
                        tooltip: {
                            text: 'Slightly Negative'
                        }
                    },
                    {
                        limit: 55,
                        showTick: true,
                        tooltip: {
                            text: 'Neutral'
                        }
                    },
                    {
                        limit: 70,
                        showTick: true,
                        tooltip: {
                            text: 'Slightly Positive'
                        }
                    },
                    {
                        limit: 85,
                        showTick: true,
                        tooltip: {
                            text: 'Positive'
                        }
                    },
                    {
                        limit: 100,
                        showTick: true,
                        tooltip: {
                            text: 'Mostly Positive'
                        }
                    },
                ]
            }}
            value={score}
            minValue={0}
            maxValue={100}
            pointer={{
                type: "arrow",
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
                    maxDecimalDigits: 0,
                },
                tickLabels: {
                    type: 'outer',
                    defaultTickValueConfig: {
                        style: {
                            fontSize: "0.75rem",
                            fill: tickLabelColor,
                            textShadow: theme === 'dark' ?
                                'black 1px 1px 0px, black 0px 0px 2.5em, black 0px 0px 0.2em' :
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

