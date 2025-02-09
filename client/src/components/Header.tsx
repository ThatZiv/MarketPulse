import React from 'react'
import { Button } from './ui/button'
import { Popover, PopoverTrigger } from './ui/popover'
import { PopoverContent } from '@radix-ui/react-popover'
import MobileMenu, { MenuItem } from './MobileMenu'
import { BookOpen, Menu } from 'lucide-react';
import {
    NavigationMenu,
    NavigationMenuContent,
    NavigationMenuIndicator,
    NavigationMenuItem,
    NavigationMenuLink,
    NavigationMenuList,
    NavigationMenuTrigger,
    NavigationMenuViewport,
    navigationMenuTriggerStyle
} from "@/components/ui/navigation-menu"


const MenuItems: MenuItem[] = [
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
            {
                href: '/get-started',
                label: 'Get Started',
                icon: <BookOpen size={24} />,
                desc: 'Get started with MarketPulse'
            },
        ]
    },
    {
        href: '#',
        label: 'Features',
        submenu: [
            {
                href: '#',
                label: 'Hype Meter',
                icon: <BookOpen size={24} />,
                desc: 'Get started with MarketPulse'
            },
        ]
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
        <header className='bg-tertiary/50 h-16 grid grid-cols-1 items-center md:h-20'>
            <div className='container flex justify-between items-center px-2 mx-auto 2xl:max-w-screen-xl lg:grid lg:grid-cols-[1fr,3fr,1fr]'>
                <h1 className='text-white'>MarketPulse</h1>
                <NavigationMenu className='max-lg:hidden mx-auto'>
                    <NavigationMenuList>
                        {MenuItems.map(({ href, label, submenu }, index) => (
                            <NavigationMenuItem key={index}>
                                {submenu ? (
                                    <>
                                        <NavigationMenuTrigger>
                                            {label}
                                        </NavigationMenuTrigger>
                                        <NavigationMenuContent>
                                            <ul className="grid grid-cols-2 gap-2 p-2 w-[40rem]">
                                                {submenu.map(({ href, label, icon, desc }, index) => (
                                                    <li key={index}>
                                                        <NavigationMenuLink asChild>
                                                            <a href={href} className='flex gap-3 select-none p-2
                                                            rounded-sm transition-colors hover:bg-foreground/5'>
                                                                <div className='w-10 h-10 bg-foreground/10 
                                                                rounded-sm shadow-sm border-t border-foreground/5
                                                                flex-shrink-0 grid place-items-center'>{icon}</div>
                                                                <div>
                                                                    <div className='text-sm leading-normal
                                                                    mb-1'>{label}</div>
                                                                    <p className='text-sm leading-normal 
                                                                    text-muted-foreground'>{desc}</p>
                                                                </div>
                                                            </a>
                                                        </NavigationMenuLink>
                                                    </li>
                                                ))}
                                            </ul>
                                        </NavigationMenuContent>
                                    </>
                                ) : (
                                    <NavigationMenuLink href={href} className={navigationMenuTriggerStyle()}>
                                        {label}
                                    </NavigationMenuLink>
                                )}

                            </NavigationMenuItem>
                        ))}
                    </NavigationMenuList>
                </NavigationMenu>
                <div className='flex items-center gap-2 justify-end max-lg:hidden'>
                    <Button variant="ghost" className='w-full text-white hover:text-black'>
                        Sign In
                    </Button>
                    <Button className='w-full'>
                        Create account
                    </Button>
                </div>
                <Popover>
                    <PopoverTrigger asChild>
                        <Button variant="outline" size="icon" className="lg:hidden border border-gray-600" >
                            <Menu size={24} />
                        </Button>
                    </PopoverTrigger>
                    <PopoverContent align='end' className='backdrop-blur-3xl border-foreground/5 border-b-0 
                    border-x-0 rounded-lg overflow-hidden'>
                        <MobileMenu menuItems={MenuItems} />
                    </PopoverContent>

                </Popover>
            </div>
        </header>
    )
}

export default Header
