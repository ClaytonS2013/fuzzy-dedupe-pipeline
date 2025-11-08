-- Fix for Supabase schema
-- Run this in Supabase SQL editor

-- Add missing columns to practice_records
ALTER TABLE practice_records 
ADD COLUMN IF NOT EXISTS canonical TEXT,
ADD COLUMN IF NOT EXISTS reasoning TEXT,
ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 0,
ADD COLUMN IF NOT EXISTS suggested_fix TEXT,
ADD COLUMN IF NOT EXISTS practice_type TEXT,
ADD COLUMN IF NOT EXISTS ai_processed BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS embedding_vector FLOAT[] DEFAULT NULL;

-- Create index for faster searches
CREATE INDEX IF NOT EXISTS idx_practice_records_canonical 
ON practice_records(canonical);

CREATE INDEX IF NOT EXISTS idx_practice_records_confidence 
ON practice_records(confidence);

-- Verify columns exist
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'practice_records'
ORDER BY ordinal_position;
