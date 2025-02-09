import React from 'react'
import { Button } from './ui/button'
import { Popover, PopoverTrigger } from './ui/popover'
import { PopoverContent } from '@radix-ui/react-popover'
import MobileMenu, { MenuItem } from './MobileMenu'
import { BookOpen, Menu } from 'lucide-react';

const MobileMenuItems: MenuItem[] = [
    {
        href: '/Documentation',
        label: 'Documentation',
        submenu: [
            {
            href: '/tutorial',
            label: 'Tutorials',
            icon: <BookOpen size={24} />,
            desc: 'Get started with MarketPulse'
            },
        ]
    },
    {
        href: '/introduction',
        label: 'Introduction',
    },
    {
        href: '/get-started',
        label: 'Get Started',
    },
    {
        href: '/auth',
        label: 'Login',
    },
    {
        href: '/signup',
        label: 'Sign Up',
    }
]
const Header = () => {
    return (
        <header className='border h-16 grid grid-cols-1 items-center md:h-20 lg:h-24'>
            <div className='container flex justify-between px-2 mx-auto 2xl:max-w-screen-xl'>
                <h1>MarketPulse</h1>
                <Popover>
                    <PopoverTrigger asChild>
                        <Button variant="outline" size="icon" className="lg:hidden border border-gray-600" >
                            <Menu size={24} />
                        </Button>
                    </PopoverTrigger>
                    <PopoverContent>
                        <MobileMenu menuItems={MobileMenuItems} />
                    </PopoverContent>

                </Popover>
            </div>
        </header>
    )
}

export default Header
