-- ============================================
-- Supabase Setup Script
-- ============================================
-- Run this SQL in your Supabase SQL Editor
-- ============================================

-- 1. Create sync_state table to track last processed data
CREATE TABLE IF NOT EXISTS sync_state (
  id INT PRIMARY KEY DEFAULT 1,
  last_processed_marker TEXT,
  last_run_at TIMESTAMPTZ,
  CONSTRAINT single_row CHECK (id = 1)
);

-- 2. Insert initial sync state record
INSERT INTO sync_state (id, last_processed_marker, last_run_at)
VALUES (1, '', NULL)
ON CONFLICT (id) DO NOTHING;

-- 3. Add external_id column to practice_records table (if not exists)
-- This column is used for upsert logic to prevent duplicates
ALTER TABLE practice_records 
ADD COLUMN IF NOT EXISTS external_id TEXT UNIQUE;

-- 4. Create index on external_id for better upsert performance
CREATE INDEX IF NOT EXISTS idx_practice_records_external_id 
ON practice_records(external_id);

-- 5. Verify tables exist
-- You should already have these tables:
--   - practice_records (raw data input)
--   - dedupe_results (cleaned data output from your AI pipeline)

-- 6. (Optional) View current sync state
SELECT * FROM sync_state;

-- ============================================
-- Setup Complete!
-- ============================================
