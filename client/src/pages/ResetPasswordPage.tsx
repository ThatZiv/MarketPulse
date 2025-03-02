import { ResetPasswordForm } from "@/components/reset_password";
import { ModeToggle } from "@/components/ui/mode-toggle";


export default function ResetPasswordPage() {
    return(<div className="m-0 p-0 min-h-screen bg-center bg-no-repeat bg-cover bg-stocks-graph dark:bg-dark-stocks-graph">
          <div className="flex items-center justify-left text-3xl font-bold text-transparent p-2">
            <span className="text-white ml-5 pt-2 mb-5">MarketPulse</span>
            <div className="absolute right-0 top-0 p-2">
              <ModeToggle />
            </div>
          </div>
          <div
            className="md:max-w-lg max-w-xs mx-auto pb-8 "
            
          >  <ResetPasswordForm/></div>
          {/* <Footer /> */}
        </div>)
}
