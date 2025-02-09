import React from 'react'
import { Button } from './ui/button';
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { ChevronsUpDown } from 'lucide-react';
import { Separator } from './ui/separator';
export type MenuItem = {
    href: string;
    label: string;
    submenu?: SubMenuItem[];
}
type SubMenuItem = {
    href: string;
    label: string;
    icon: JSX.Element;
    desc: string;
}
type MobileMenuProps = {
    menuItems: MenuItem[];
}
const MobileMenu = ({ menuItems }: MobileMenuProps) => {
    return (
        <div className="bg-black text-white rounded-lg pb-3">
            <ul>
                {menuItems.map(({ href, label, submenu }, index) => (
                    <li key={index} className='w-full' >
                        {submenu ? (
                            <Collapsible>
                                <CollapsibleTrigger className="w-full">
                                    <Button asChild variant="ghost" className='w-full pr-3 justify-start'>
                                        <div className='w-full gap-10 justify-between px-2'>
                                            {label}
                                            <ChevronsUpDown className=''/>
                                        </div>

                                    </Button>
                                </CollapsibleTrigger>
                                <CollapsibleContent className="ps-2 ">
                                    <ul className="border-l border-l-muted-foreground/60">
                                        {submenu.map(({ href, label }, index) => (
                                            <li key={index}>
                                                <Button asChild variant="ghost" className='w-full justify-start text-muted-foreground
                                                hover:bg-transparent hover:text-white'>
                                                    <a href={href}>{label}</a>
                                                </Button>
                                            </li>
                                        ))}
                                    </ul>
                                </CollapsibleContent>
                            </Collapsible>
                        ) : (
                            <Button asChild variant="ghost" className='w-full justify-start'>
                                <a href={href}>{label}</a>
                            </Button>
                        )}
                    </li>
                ))}
            </ul>
            <Separator className='bg-muted-foreground/60'/>
            <div className='flex items-center gap-2 mt-4 mx-4'>
                <Button variant="ghost" className='w-full'>
                    Sign In
                </Button>
                <Button className='w-full'>
                    Create account
                </Button>
            </div>
        </div>
    )
}

export default MobileMenu
