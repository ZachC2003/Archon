### Financial Analysis Extension (Optional)
To enable financial analysis persistence in Supabase:
1. Ensure `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set in your `.env` file.
2. Run the following SQL in your Supabase SQL Editor:
   - Copy the contents of `utils/financial_analysis.sql`.
   - Paste and execute it in the Supabase dashboard (https://supabase.com/dashboard/project/<your-project-id>/sql).
3. The `mcp_server.py` will automatically use this table if configured.
Note: This is not required for Archon’s core functionality and is only needed for financial features via MCP.