alter table "public"."User_Stocks" drop constraint "User_Stocks_user_id_fkey";

alter table "public"."User_Stocks" drop constraint "User_Stocks_pkey";

drop index if exists "public"."User_Stocks_pkey";

alter table "public"."Account" alter column "user_id" set default auth.uid();

alter table "public"."Account" alter column "user_id" set data type uuid using "user_id"::uuid;

alter table "public"."Account" enable row level security;

alter table "public"."Stock_Info" enable row level security;

alter table "public"."Stocks" enable row level security;

alter table "public"."User_Stocks" drop column "amount_owned";

alter table "public"."User_Stocks" add column "desired_investiture" real not null;

alter table "public"."User_Stocks" add column "shares_owned" double precision not null default '0'::double precision;

alter table "public"."User_Stocks" alter column "user_id" set default auth.uid();

alter table "public"."User_Stocks" alter column "user_id" set data type uuid using "user_id"::uuid;

alter table "public"."User_Stocks" enable row level security;

drop sequence if exists "public"."Account_user_id_seq";

CREATE UNIQUE INDEX "User_Stocks_user_id_key" ON public."User_Stocks" USING btree (user_id);

CREATE UNIQUE INDEX "User_Stocks_pkey" ON public."User_Stocks" USING btree (stock_id, user_id);

alter table "public"."User_Stocks" add constraint "User_Stocks_pkey" PRIMARY KEY using index "User_Stocks_pkey";

alter table "public"."Account" add constraint "Account_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE CASCADE not valid;

alter table "public"."Account" validate constraint "Account_user_id_fkey";

alter table "public"."User_Stocks" add constraint "User_Stocks_desired_investiture_check" CHECK ((desired_investiture > (0)::double precision)) not valid;

alter table "public"."User_Stocks" validate constraint "User_Stocks_desired_investiture_check";

alter table "public"."User_Stocks" add constraint "User_Stocks_user_id_key" UNIQUE using index "User_Stocks_user_id_key";

alter table "public"."User_Stocks" add constraint "User_Stocks_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE CASCADE not valid;

alter table "public"."User_Stocks" validate constraint "User_Stocks_user_id_fkey";

create policy "Let users do anything they want with their own account"
on "public"."Account"
as permissive
for all
to public
using ((auth.uid() = user_id));


create policy "Enable read access for all users"
on "public"."Stock_Info"
as permissive
for select
to authenticated
using (true);


create policy "Enable read access for all users"
on "public"."Stocks"
as permissive
for select
to authenticated
using (true);


create policy "Let users do anything to their own stocks"
on "public"."User_Stocks"
as permissive
for all
to authenticated
using ((auth.uid() = user_id))
with check ((auth.uid() = user_id));



