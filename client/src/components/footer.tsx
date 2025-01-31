import React from 'react'

const items = ['Docs', 'Contact Us', 'About Us', 'FAQs'];
export function Footer() {
    return (
        <div className='flex items-center justify-center mb-5'>
            <ul className='grid grid-cols-4 gap-5'>
                {items.map((item, index) => (
                    <li key={index}>{item}</li> //will convert to links later or change the style entirely. This 
                    // only a base for footer.
                ))}
            </ul>
        </div>
    )
}

export default Footer
