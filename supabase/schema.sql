

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


CREATE EXTENSION IF NOT EXISTS "pgsodium" WITH SCHEMA "pgsodium";






COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";






CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgjwt" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";






CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";





SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."Account" (
    "first_name" character varying,
    "last_name" character varying,
    "user_id" "uuid" DEFAULT "auth"."uid"() NOT NULL
);


ALTER TABLE "public"."Account" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."Stock_Info" (
    "stock_id" integer NOT NULL,
    "stock_close" double precision,
    "stock_volume" bigint,
    "stock_open" double precision,
    "stock_high" double precision,
    "stock_low" double precision,
    "sentiment_data" double precision,
    "time_stamp" "date" NOT NULL
);


ALTER TABLE "public"."Stock_Info" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."Stocks" (
    "stock_id" integer NOT NULL,
    "stock_ticker" character varying,
    "stock_name" character varying
);


ALTER TABLE "public"."Stocks" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."Stocks_stock_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."Stocks_stock_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."Stocks_stock_id_seq" OWNED BY "public"."Stocks"."stock_id";



CREATE TABLE IF NOT EXISTS "public"."User_Stocks" (
    "stock_id" integer NOT NULL,
    "shares_owned" double precision DEFAULT '0'::double precision NOT NULL,
    "desired_investiture" real NOT NULL,
    "user_id" "uuid" DEFAULT "auth"."uid"() NOT NULL,
    "created_at" "date",
    CONSTRAINT "User_Stocks_desired_investiture_check" CHECK (("desired_investiture" > (0)::double precision))
);


ALTER TABLE "public"."User_Stocks" OWNER TO "postgres";


COMMENT ON COLUMN "public"."User_Stocks"."desired_investiture" IS 'how much do they want to invest';



ALTER TABLE "public"."User_Stocks" ALTER COLUMN "stock_id" ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME "public"."User_Stocks_stock_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



ALTER TABLE ONLY "public"."Stocks" ALTER COLUMN "stock_id" SET DEFAULT "nextval"('"public"."Stocks_stock_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."Account"
    ADD CONSTRAINT "Account_pkey" PRIMARY KEY ("user_id");



ALTER TABLE ONLY "public"."Stock_Info"
    ADD CONSTRAINT "Stock_Info_pkey" PRIMARY KEY ("stock_id", "time_stamp");



ALTER TABLE ONLY "public"."Stocks"
    ADD CONSTRAINT "Stocks_pkey" PRIMARY KEY ("stock_id");



ALTER TABLE ONLY "public"."User_Stocks"
    ADD CONSTRAINT "User_Stocks_pkey" PRIMARY KEY ("stock_id", "user_id");



ALTER TABLE ONLY "public"."User_Stocks"
    ADD CONSTRAINT "User_Stocks_stock_id_key" UNIQUE ("stock_id");



ALTER TABLE ONLY "public"."Account"
    ADD CONSTRAINT "Account_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON UPDATE CASCADE ON DELETE CASCADE;



ALTER TABLE ONLY "public"."Stock_Info"
    ADD CONSTRAINT "Stock_Info_stock_id_fkey" FOREIGN KEY ("stock_id") REFERENCES "public"."Stocks"("stock_id");



ALTER TABLE ONLY "public"."User_Stocks"
    ADD CONSTRAINT "User_Stocks_stock_id_fkey" FOREIGN KEY ("stock_id") REFERENCES "public"."Stocks"("stock_id");



ALTER TABLE ONLY "public"."User_Stocks"
    ADD CONSTRAINT "User_Stocks_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON UPDATE CASCADE ON DELETE CASCADE;



ALTER TABLE "public"."Account" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "Enable read access for all users" ON "public"."Stock_Info" FOR SELECT TO "authenticated" USING (true);



CREATE POLICY "Enable read access for all users" ON "public"."Stocks" FOR SELECT TO "authenticated" USING (true);



CREATE POLICY "Let users do anything they want with their own account" ON "public"."Account" USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Let users do anything to their own stocks" ON "public"."User_Stocks" TO "authenticated" USING (("auth"."uid"() = "user_id")) WITH CHECK (("auth"."uid"() = "user_id"));



ALTER TABLE "public"."Stock_Info" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."Stocks" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."User_Stocks" ENABLE ROW LEVEL SECURITY;




ALTER PUBLICATION "supabase_realtime" OWNER TO "postgres";


GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";



































































































































































































GRANT ALL ON TABLE "public"."Account" TO "anon";
GRANT ALL ON TABLE "public"."Account" TO "authenticated";
GRANT ALL ON TABLE "public"."Account" TO "service_role";



GRANT ALL ON TABLE "public"."Stock_Info" TO "anon";
GRANT ALL ON TABLE "public"."Stock_Info" TO "authenticated";
GRANT ALL ON TABLE "public"."Stock_Info" TO "service_role";



GRANT ALL ON TABLE "public"."Stocks" TO "anon";
GRANT ALL ON TABLE "public"."Stocks" TO "authenticated";
GRANT ALL ON TABLE "public"."Stocks" TO "service_role";



GRANT ALL ON SEQUENCE "public"."Stocks_stock_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."Stocks_stock_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."Stocks_stock_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."User_Stocks" TO "anon";
GRANT ALL ON TABLE "public"."User_Stocks" TO "authenticated";
GRANT ALL ON TABLE "public"."User_Stocks" TO "service_role";



GRANT ALL ON SEQUENCE "public"."User_Stocks_stock_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."User_Stocks_stock_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."User_Stocks_stock_id_seq" TO "service_role";



ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "service_role";






























RESET ALL;
