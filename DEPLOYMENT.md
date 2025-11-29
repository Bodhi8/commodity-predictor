# 🚀 Deployment Guide: Automated Commodity Predictions

This guide covers all options for automating your commodity predictions and displaying them on your website.

---

## Quick Comparison

| Option | Cost | Setup Time | Reliability | Best For |
|--------|------|------------|-------------|----------|
| **GitHub Actions** | Free | 15 min | ⭐⭐⭐⭐⭐ | Most users |
| **Google Colab** | Free | 10 min | ⭐⭐⭐ | Manual/occasional runs |
| **Local + Cron** | Free | 20 min | ⭐⭐⭐ | Tech-savvy users |
| **AWS Lambda** | ~$1/mo | 45 min | ⭐⭐⭐⭐⭐ | Enterprise |

---

## Option 1: GitHub Actions (Recommended) ✅

**Cost: FREE | Reliability: Excellent | Maintenance: None**

### Step 1: Create GitHub Repository

```bash
# Create new repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/commodity-predictor.git
cd commodity-predictor

# Copy all project files
cp -r /path/to/commodity_predictor/* .

# Push to GitHub
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2: Enable GitHub Actions

1. Go to your repo → **Settings** → **Actions** → **General**
2. Select "Allow all actions"
3. Under "Workflow permissions", select "Read and write permissions"
4. Click Save

### Step 3: Enable GitHub Pages (for hosting dashboard)

1. Go to **Settings** → **Pages**
2. Source: "Deploy from a branch"
3. Branch: `main` / `root`
4. Click Save

### Step 4: Update Dashboard URL

Edit `dashboard.html` line 180:
```javascript
const PREDICTIONS_URL = 'https://YOUR_USERNAME.github.io/commodity-predictor/predictions/latest_predictions.json';
```

### Step 5: Trigger First Run

1. Go to **Actions** tab
2. Click "Daily Commodity Predictions"
3. Click "Run workflow"

### Done! 

- **Dashboard URL**: `https://YOUR_USERNAME.github.io/commodity-predictor/dashboard.html`
- **Predictions run**: Daily at 6 AM UTC automatically
- **Embed on your site**: See "Website Integration" below

---

## Option 2: Google Colab

**Cost: FREE | Best for: Manual runs or Colab Pro scheduling**

### Step 1: Upload Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Upload `Commodity_Predictions_Colab.ipynb`
3. Run all cells

### Step 2: Download Results

The notebook will download `latest_predictions.json` automatically.

### Step 3: Host the JSON

Upload to one of:
- **GitHub Gist** (free, easy)
- **Google Drive** (free, share link)
- **Your web server**

### Automation with Colab Pro

Colab Pro ($10/mo) includes scheduled notebooks:
1. Click the clock icon in the toolbar
2. Set schedule (e.g., daily at 6 AM)

---

## Option 3: Local Machine + Cron

**Cost: FREE | Requires: Computer always on**

### Windows (Task Scheduler)

1. Open Task Scheduler
2. Create Basic Task → "Commodity Predictions"
3. Trigger: Daily at your preferred time
4. Action: Start a program
   - Program: `python`
   - Arguments: `C:\path\to\daily_predictions.py`
5. Finish

### Mac/Linux (Cron)

```bash
# Edit crontab
crontab -e

# Add this line (runs at 6 AM daily)
0 6 * * * cd /path/to/commodity_predictor && python daily_predictions.py >> /var/log/commodity.log 2>&1
```

### Auto-upload to GitHub

Add to the end of your cron job:
```bash
0 6 * * * cd /path/to/commodity_predictor && python daily_predictions.py && git add predictions/ && git commit -m "Update" && git push
```

---

## Website Integration

### Option A: Full Dashboard (Standalone Page)

Host `dashboard.html` on your website. Update the PREDICTIONS_URL:

```javascript
const PREDICTIONS_URL = 'https://YOUR_USERNAME.github.io/commodity-predictor/predictions/latest_predictions.json';
```

### Option B: Embed Widget

Add this to any page on your website:

```html
<!-- Commodity Predictions Widget -->
<div id="commodity-widget"></div>
<script>
const PREDICTIONS_URL = 'https://YOUR_USERNAME.github.io/commodity-predictor/predictions/latest_predictions.json';

// ... (copy content from embed_widget.html)
</script>
```

### Option C: Custom Integration (React/Vue/etc.)

Fetch the JSON and build your own UI:

```javascript
// React example
import { useState, useEffect } from 'react';

function CommodityPredictions() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    fetch('https://YOUR_USERNAME.github.io/commodity-predictor/predictions/latest_predictions.json')
      .then(r => r.json())
      .then(setData);
  }, []);
  
  if (!data) return <div>Loading...</div>;
  
  return (
    <div>
      {Object.entries(data.predictions).map(([name, info]) => (
        <div key={name}>
          <h3>{name}</h3>
          <p>Current: ${info.current_price}</p>
          <p>1-Day Forecast: ${info.forecasts['1d'].price} 
             ({info.forecasts['1d'].change_pct > 0 ? '+' : ''}{info.forecasts['1d'].change_pct}%)
          </p>
        </div>
      ))}
    </div>
  );
}
```

### Option D: iframe Embed

Simplest option - just embed the dashboard:

```html
<iframe 
  src="https://YOUR_USERNAME.github.io/commodity-predictor/dashboard.html" 
  width="100%" 
  height="800" 
  frameborder="0">
</iframe>
```

---

## JSON Output Format

The `latest_predictions.json` file has this structure:

```json
{
  "generated_at": "2025-01-15T06:00:00.000Z",
  "predictions": {
    "Crude Oil WTI": {
      "ticker": "CL=F",
      "current_price": 72.45,
      "last_updated": "2025-01-14",
      "forecasts": {
        "1d": { "price": 73.12, "change_pct": 0.92 },
        "5d": { "price": 74.50, "change_pct": 2.83 },
        "10d": { "price": 75.20, "change_pct": 3.79 }
      }
    },
    "Gold": { ... },
    ...
  },
  "historical": {
    "Crude Oil WTI": {
      "dates": ["2024-10-15", "2024-10-16", ...],
      "prices": [70.50, 71.20, ...]
    },
    ...
  }
}
```

---

## Troubleshooting

### GitHub Actions not running?
1. Check Actions are enabled in Settings
2. Check workflow file is in `.github/workflows/`
3. Look at the Actions tab for error logs

### Predictions look wrong?
1. Check if market was closed (holidays/weekends)
2. Verify data is being fetched (check logs)
3. Model may need more training data

### CORS errors on website?
The JSON must be served with proper headers. GitHub Pages handles this automatically.

### Want to add more commodities?
Edit the `COMMODITIES` dict in `daily_predictions.py`:
```python
COMMODITIES = {
    'CL=F': 'Crude Oil WTI',
    'GC=F': 'Gold',
    # Add more tickers here
    'PL=F': 'Platinum',
}
```

---

## Support

- **Issues**: Open a GitHub issue
- **Commodity tickers**: [Yahoo Finance](https://finance.yahoo.com/commodities)
- **GitHub Actions docs**: [docs.github.com/actions](https://docs.github.com/actions)
