CREATE TABLE financial_analysis (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ticker TEXT NOT NULL,
    analysis_type TEXT NOT NULL CHECK (analysis_type IN ('metrics', 'sentiment', 'institutional')),
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(ticker, analysis_type)
);

CREATE INDEX idx_financial_analysis_ticker ON financial_analysis(ticker);