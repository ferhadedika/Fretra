-- Tightens docs/sql/enable_rls_permissive.sql's deliberately-permissive
-- interim policy into the originally-envisioned split: the public anon
-- key (the one shipped in the browser bundle, and used for the browser's
-- own direct reads of scfi_weekly via useScfi.js/useBaselines.js) gets
-- READ-ONLY access; the backend/scraper get their write access from a
-- service-role key instead, which bypasses RLS entirely by Postgres/
-- Supabase design - so once it's in use, these tables need no write
-- policy for anon/authenticated at all.
--
-- ============================================================
-- DO NOT RUN THIS until BOTH of the following are true, in order:
--
--   1. You've added SUPABASE_SERVICE_ROLE_KEY to backend/.env (get it
--      from Supabase dashboard > Project Settings > API > service_role
--      secret - NOT the anon/publishable key already in SUPABASE_KEY),
--      redeployed/restarted the backend, and confirmed the startup log
--      line reads "Supabase connected (using service-role key)" - see
--      app/services/supabase_service.py.
--
--   2. You've confirmed the scraper (scraper_pipeline.py) and any other
--      standalone script that writes to these tables directly (e.g.
--      supabase_upload.py) is ALSO updated to use the service-role key
--      if it runs anywhere outside the backend process.
--
-- If you run this SQL before the backend is actually authenticating with
-- the service-role key, its writes (the rolling-forecast pipeline,
-- accuracy backfill, the scraper) will start failing silently - each
-- Supabase call in supabase_service.py already catches exceptions and
-- only logs a warning, so nothing crashes, but the live rolling forecast
-- will simply stop updating with no loud error.
-- ============================================================

drop policy "anon+authenticated full access" on forecast_history;
drop policy "anon+authenticated full access" on forecast_shap;
drop policy "anon+authenticated full access" on forecast_accuracy;
drop policy "anon+authenticated full access" on pipeline_locks;
drop policy "anon+authenticated full access" on scfi_weekly;

-- forecast_history / forecast_shap / forecast_accuracy / pipeline_locks:
-- the browser never queries these directly (only through the FastAPI
-- backend, which now uses the service-role key and so bypasses RLS) -
-- so with RLS enabled and no policy at all, they become fully
-- inaccessible to anon/authenticated. No read policy needed either.

-- scfi_weekly: the ONE table the browser queries directly (useScfi.js /
-- useBaselines.js), and only from inside the authenticated dashboard
-- (App.jsx gates that route behind login) - read-only, authenticated
-- only, is enough for how it's actually used today.
create policy "authenticated read-only" on scfi_weekly
  for select to authenticated using (true);
