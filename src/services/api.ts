import {
  PredictionRequest,
  PredictionResponse,
  AnalyticsSummary,
  ModelMetrics,
  FeatureImportanceItem,
  MLModelType,
} from '../types/estate';
import { mlEngine, MODEL_COLUMNS } from './mlEngine';

const FASTAPI_STORAGE_KEY = 'estatiq_fastapi_url';

export function getConfiguredFastApiUrl(): string {
  try {
    return localStorage.getItem(FASTAPI_STORAGE_KEY) || '';
  } catch {
    return '';
  }
}

export function setConfiguredFastApiUrl(url: string): void {
  try {
    if (url.trim()) {
      localStorage.setItem(FASTAPI_STORAGE_KEY, url.trim().replace(/\/+$/, ''));
    } else {
      localStorage.removeItem(FASTAPI_STORAGE_KEY);
    }
  } catch {
    // Ignore storage error
  }
}

/**
 * POST /api/predict
 * Preserves the exact input feature names and categorical encoding expected by the trained models:
 * area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating,
 * airconditioning, parking, prefarea, furnishingstatus, model
 */
export async function predictPropertyPrice(
  request: PredictionRequest
): Promise<PredictionResponse> {
  const customFastApi = getConfiguredFastApiUrl();
  const endpoint = customFastApi ? `${customFastApi}/api/predict` : '/api/predict';

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        area: Number(request.area),
        bedrooms: Number(request.bedrooms),
        bathrooms: Number(request.bathrooms),
        stories: Number(request.stories),
        mainroad: request.mainroad,
        guestroom: request.guestroom,
        basement: request.basement,
        hotwaterheating: request.hotwaterheating,
        airconditioning: request.airconditioning,
        parking: Number(request.parking),
        prefarea: request.prefarea,
        furnishingstatus: request.furnishingstatus,
        model: request.model,
      }),
    });

    if (response.ok) {
      const data = await response.json();
      // If external FastAPI returns minimal response { success, predicted_price, model, currency }, enrich with local feature contributions
      if (typeof data.predicted_price === 'number') {
        const fallbackCalc = mlEngine.predict(request, request.model, 'fastapi_backend');
        return {
          ...fallbackCalc,
          ...data,
          execution_mode: customFastApi ? 'fastapi_backend' : (data.execution_mode || 'trained_ml_engine'),
        };
      }
    }
  } catch {
    // Graceful fallback to in-memory trained OLS / CART engine
  }

  // Simulate realistic 350ms model inference latency for smooth UI feedback
  await new Promise((resolve) => setTimeout(resolve, 350));
  return mlEngine.predict(request, request.model, 'trained_ml_engine');
}

/**
 * GET /api/analytics
 */
export async function fetchAnalyticsSummary(): Promise<AnalyticsSummary> {
  const customFastApi = getConfiguredFastApiUrl();
  const endpoint = customFastApi ? `${customFastApi}/api/analytics` : '/api/analytics';

  try {
    const res = await fetch(endpoint);
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Fallback to trained engine summary
  }
  return mlEngine.analyticsSummary;
}

/**
 * GET /api/model-performance
 */
export async function fetchModelPerformance(): Promise<Record<MLModelType, ModelMetrics>> {
  const customFastApi = getConfiguredFastApiUrl();
  const endpoint = customFastApi
    ? `${customFastApi}/api/model-performance`
    : '/api/model-performance';

  try {
    const res = await fetch(endpoint);
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Fallback to trained engine metrics
  }
  return mlEngine.metrics;
}

/**
 * GET /api/feature-importance
 */
export async function fetchFeatureImportance(): Promise<FeatureImportanceItem[]> {
  const customFastApi = getConfiguredFastApiUrl();
  const endpoint = customFastApi
    ? `${customFastApi}/api/feature-importance`
    : '/api/feature-importance';

  try {
    const res = await fetch(endpoint);
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Fallback
  }
  return mlEngine.featureImportanceList;
}

/**
 * GET /api/health
 */
export async function fetchHealthStatus(): Promise<{
  status: string;
  mode: string;
  models: string[];
  modelColumns: readonly string[];
  datasetRecords: number;
}> {
  const customFastApi = getConfiguredFastApiUrl();
  const endpoint = customFastApi ? `${customFastApi}/api/health` : '/api/health';

  try {
    const res = await fetch(endpoint);
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Fallback
  }
  return {
    status: 'healthy',
    mode: 'Trained ML Engine (Housing.csv 545 records)',
    models: ['linear_regression', 'decision_tree'],
    modelColumns: MODEL_COLUMNS,
    datasetRecords: mlEngine.analyticsSummary.totalRecords,
  };
}


