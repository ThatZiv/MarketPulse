import React from 'react'
import { Button } from './ui/button';
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { ChevronsUpDown } from 'lucide-react';

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
        <div>
            <ul className="">
                {menuItems.map(({ href, label, submenu }, index) => (
                    <li key={index}>
                        {submenu ? (
                            <Collapsible>
                                <CollapsibleTrigger>
                                    <Button variant="ghost" className='w-full justify-start'>
                                        {label}
                                        <ChevronsUpDown />
                                    </Button>
                                </CollapsibleTrigger>
                                <CollapsibleContent className="ps-2 ">
                                  <ul className="border-l border-l-muted-foreground/20">
                                    {submenu.map(({href,label},index)=>(
                                        <li key={index}>
                                            <Button asChild variant="ghost" className='w-full justify-start text-muted-foreground
                                            hover:bg-transparent'>
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
        </div>
    )
}

export default MobileMenu
