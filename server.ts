import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import { mlEngine, MODEL_COLUMNS } from './src/services/mlEngine.ts';
import { MLModelType, PropertyInput } from './src/types/estate.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // GET /api/health
  app.get('/api/health', (_req, res) => {
    res.json({
      status: 'healthy',
      mode: process.env.FASTAPI_URL
        ? 'FastAPI Bridge Active'
        : 'Trained ML Engine (Housing.csv 545 records)',
      models: ['linear_regression', 'decision_tree'],
      modelColumns: MODEL_COLUMNS,
      datasetRecords: mlEngine.analyticsSummary.totalRecords,
    });
  });

  // POST /api/predict
  app.post('/api/predict', async (req, res) => {
    try {
      const body = req.body;
      const model: MLModelType =
        body.model === 'decision_tree' ? 'decision_tree' : 'linear_regression';

      const input: PropertyInput = {
        area: Number(body.area) || 5000,
        bedrooms: Number(body.bedrooms) || 3,
        bathrooms: Number(body.bathrooms) || 2,
        stories: Number(body.stories) || 2,
        mainroad: body.mainroad === 'no' ? 'no' : 'yes',
        guestroom: body.guestroom === 'yes' ? 'yes' : 'no',
        basement: body.basement === 'yes' ? 'yes' : 'no',
        hotwaterheating: body.hotwaterheating === 'yes' ? 'yes' : 'no',
        airconditioning: body.airconditioning === 'yes' ? 'yes' : 'no',
        parking: Number(body.parking) ?? 1,
        prefarea: body.prefarea === 'yes' ? 'yes' : 'no',
        furnishingstatus:
          body.furnishingstatus === 'furnished' ||
          body.furnishingstatus === 'semi-furnished' ||
          body.furnishingstatus === 'unfurnished'
            ? body.furnishingstatus
            : 'semi-furnished',
      };

      // Optional proxy to external Python FastAPI service if FASTAPI_URL env is set
      if (process.env.FASTAPI_URL) {
        try {
          const fastRes = await fetch(`${process.env.FASTAPI_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...input, model }),
          });
          if (fastRes.ok) {
            const fastData = await fastRes.json();
            const enriched = mlEngine.predict(input, model, 'fastapi_backend');
            res.json({
              ...enriched,
              ...fastData,
              execution_mode: 'fastapi_backend',
            });
            return;
          }
        } catch {
          // Fall through to built-in trained engine
        }
      }

      const result = mlEngine.predict(input, model, 'trained_ml_engine');
      res.json(result);
    } catch (error) {
      res.status(400).json({
        success: false,
        error: error instanceof Error ? error.message : 'Prediction failed',
      });
    }
  });

  // GET /api/analytics
  app.get('/api/analytics', (_req, res) => {
    res.json(mlEngine.analyticsSummary);
  });

  // GET /api/model-performance
  app.get('/api/model-performance', (_req, res) => {
    res.json(mlEngine.metrics);
  });

  // GET /api/feature-importance
  app.get('/api/feature-importance', (_req, res) => {
    res.json(mlEngine.featureImportanceList);
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`ESTATIQ AI Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
