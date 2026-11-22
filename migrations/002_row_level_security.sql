-- ============================================================
-- PlantCare AI — Migration 002: Row Level Security & Access Control
-- ============================================================

-- 1. Enable Row Level Security (RLS) on all domain tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.diseases ENABLE ROW LEVEL SECURITY;

-- 2. Trigger Function: Prevent Privilege Escalation
-- Normal users cannot update their own role or promote themselves to 'admin'.
CREATE OR REPLACE FUNCTION public.prevent_role_escalation()
RETURNS TRIGGER AS $$
BEGIN
    -- If role is being changed to 'admin'
    IF NEW.role = 'admin' AND (OLD.role IS NULL OR OLD.role <> 'admin') THEN
        -- Allow only service_role (backend/API) or superuser
        IF current_setting('request.jwt.claim.role', true) <> 'service_role' 
           AND current_user NOT IN ('postgres', 'service_role', 'supabase_admin') THEN
            RAISE EXCEPTION 'Access Denied: Normal users are not permitted to grant administrator privileges.';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS trg_prevent_role_escalation ON public.users;
CREATE TRIGGER trg_prevent_role_escalation
BEFORE INSERT OR UPDATE OF role ON public.users
FOR EACH ROW EXECUTE FUNCTION public.prevent_role_escalation();

-- 3. Row Level Security Policies for Users Table
DROP POLICY IF EXISTS "Users can view their own profile or service_role" ON public.users;
CREATE POLICY "Users can view their own profile or service_role"
ON public.users FOR SELECT
USING (
    auth.uid()::text = id::text 
    OR current_setting('request.jwt.claim.role', true) = 'service_role'
    OR current_user IN ('postgres', 'service_role')
);

DROP POLICY IF EXISTS "Users can update their own profile" ON public.users;
CREATE POLICY "Users can update their own profile"
ON public.users FOR UPDATE
USING (
    auth.uid()::text = id::text 
    OR current_setting('request.jwt.claim.role', true) = 'service_role'
);

DROP POLICY IF EXISTS "Service role and anon registration can insert users" ON public.users;
CREATE POLICY "Service role and anon registration can insert users"
ON public.users FOR INSERT
WITH CHECK (true);

-- 4. Row Level Security Policies for Predictions Table (Data Isolation)
DROP POLICY IF EXISTS "Users can only view their own predictions" ON public.predictions;
CREATE POLICY "Users can only view their own predictions"
ON public.predictions FOR SELECT
USING (
    auth.uid()::text = user_id::text 
    OR current_setting('request.jwt.claim.role', true) = 'service_role'
    OR current_user IN ('postgres', 'service_role')
);

DROP POLICY IF EXISTS "Users can insert their own predictions" ON public.predictions;
CREATE POLICY "Users can insert their own predictions"
ON public.predictions FOR INSERT
WITH CHECK (
    auth.uid()::text = user_id::text 
    OR current_setting('request.jwt.claim.role', true) = 'service_role'
    OR current_user IN ('postgres', 'service_role')
);

DROP POLICY IF EXISTS "Users can delete their own predictions" ON public.predictions;
CREATE POLICY "Users can delete their own predictions"
ON public.predictions FOR DELETE
USING (
    auth.uid()::text = user_id::text 
    OR current_setting('request.jwt.claim.role', true) = 'service_role'
);

-- 5. Row Level Security Policies for Disease Library Table
DROP POLICY IF EXISTS "Public read access to disease knowledge library" ON public.diseases;
CREATE POLICY "Public read access to disease knowledge library"
ON public.diseases FOR SELECT
USING (true);

DROP POLICY IF EXISTS "Only administrators or service role can modify disease library" ON public.diseases;
CREATE POLICY "Only administrators or service role can modify disease library"
ON public.diseases FOR ALL
USING (
    current_setting('request.jwt.claim.role', true) = 'service_role'
    OR current_user IN ('postgres', 'service_role')
);

-- 6. Private Storage Bucket for Leaf Uploads & Diagnostic Reports
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'plantcare-uploads',
    'plantcare-uploads',
    false,
    10485760, -- 10MB limit matching security_utils.MAX_UPLOAD_SIZE_BYTES
    ARRAY['image/jpeg', 'image/png', 'application/pdf']
)
ON CONFLICT (id) DO NOTHING;

-- Storage RLS: Users can access only their own subfolder: {user_id}/*
DROP POLICY IF EXISTS "Users can upload to their own folder" ON storage.objects;
CREATE POLICY "Users can upload to their own folder"
ON storage.objects FOR INSERT
WITH CHECK (
    bucket_id = 'plantcare-uploads'
    AND (
        (storage.foldername(name))[1] = auth.uid()::text
        OR current_setting('request.jwt.claim.role', true) = 'service_role'
    )
);

DROP POLICY IF EXISTS "Users can view files in their own folder" ON storage.objects;
CREATE POLICY "Users can view files in their own folder"
ON storage.objects FOR SELECT
USING (
    bucket_id = 'plantcare-uploads'
    AND (
        (storage.foldername(name))[1] = auth.uid()::text
        OR current_setting('request.jwt.claim.role', true) = 'service_role'
    )
);
