-- Distributed lock table so only one backend instance runs the rolling
-- forecast pipeline at a time, even if two processes/instances are up.
-- Acquisition logic (in Python) is a two-phase INSERT-else-conditional-
-- UPDATE, both of which are atomic single Postgres statements, so this
-- table alone (no custom SQL function) is enough for correctness - see
-- app/services/supabase_service.py's try_acquire_lock()/release_lock().

create table pipeline_locks (
  lock_name   text primary key,
  locked_by   text not null,
  locked_at   timestamptz not null default now(),
  expires_at  timestamptz not null
);

alter table pipeline_locks enable row level security;
create policy "anon+authenticated full access" on pipeline_locks
  for all to anon, authenticated using (true) with check (true);
