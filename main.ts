// free-model-aggregator-01 entry: one deterministic aggregator server.
//
// Reads the brief ($MAKEFUN_PROMPT_FILE: provider directory, quality table,
// workload manifest), probes every provider's catalog route, registers one
// catalog with the env, then serves the client's requests on /v1/chat/completions,
// /v1/messages and /v1/models — picking the best free model that meets each
// request's requirements, speaking the provider's wire (openai, anthropic or
// gemini), and failing over down the rating order when a provider refuses.
//
// Determinism: no wall clock, no randomness; every choice is a function of the
// brief, the requests and the providers' answers. Node 22 stdlib only.
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const SEED = process.env.MAKEFUN_TRIAL_SEED ?? '';
const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const CALL_TIMEOUT_MS = 20000;
const REGISTER_RETRIES = 20;
// The /v1/models answer and register bodies carry this fixed timestamp:
// nothing in the entry may depend on the wall clock.
const FIXED_CREATED = 1767225600;

type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic' | 'gemini';
type Auth = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
interface ProviderDir { id: string; name: string; wire: Wire; auth: Auth; key_env: string; key: string; path: string; }
interface RatingRow { model: string; provider: string; rating_milli: number; context_length: number; tools: boolean; vision: boolean; json: boolean; free: boolean; }
interface ManifestReq { id: string; api: 'openai' | 'anthropic'; stream: boolean; weight: number; requires: { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number }; }
interface Brief { providers: ProviderDir[]; ratings: RatingRow[]; manifest: ManifestReq[]; }
interface CatalogModel { id: string; free: boolean; context_length: number | null; }
interface CatalogProvider { id: string; base_url: string; key_env: string; wire: Wire; models: CatalogModel[]; }
interface Catalog { providers: CatalogProvider[]; }

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
function log(message: string): void { process.stderr.write(`aggregator: ${message}\n`); }

// ---------- the brief ----------
function jsonDocs(text: string): Json[] {
  const out: Json[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('{"schema":')) {
      try {
        const value: unknown = JSON.parse(trimmed);
        if (isRecord(value)) out.push(value);
      } catch { /* not one of ours */ }
    }
  }
  return out;
}

function readBrief(file: string): Brief | null {
  if (file === '') return null;
  let text = '';
  try { text = readFileSync(file, 'utf8'); } catch { return null; }
  const docs = jsonDocs(text);
  let providers: ProviderDir[] = [];
  let ratings: RatingRow[] = [];
  let manifest: ManifestReq[] = [];
  for (const doc of docs) {
    const schema = typeof doc.schema === 'string' ? doc.schema : '';
    if (schema.endsWith('/providers/v1') && Array.isArray(doc.providers)) providers = doc.providers as ProviderDir[];
    if (schema.endsWith('/ratings/v1') && Array.isArray(doc.ratings)) ratings = doc.ratings as RatingRow[];
    if (schema.endsWith('/workload/v1') && Array.isArray(doc.manifest)) manifest = doc.manifest as ManifestReq[];
  }
  if (providers.length === 0) return null;
  return { providers, ratings, manifest };
}

// ---------- the catalog ----------
function authHeaders(provider: ProviderDir): Json {
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

function catalogRouteOf(wire: Wire): string {
  return wire === 'gemini' ? '/v1beta/models' : '/v1/models';
}

interface ProbedProvider { provider: ProviderDir; ok: boolean; models: CatalogModel[]; }

async function probeProvider(provider: ProviderDir): Promise<ProbedProvider> {
  const url = `${ENV_URL}${provider.path}${catalogRouteOf(provider.wire)}`;
  try {
    const response = await fetch(url, { headers: { ...authHeaders(provider), accept: 'application/json' }, signal: AbortSignal.timeout(8000) });
    if (!response.ok) return { provider, ok: false, models: [] };
    const body: unknown = await response.json();
    const models: CatalogModel[] = [];
    if (!isRecord(body)) return { provider, ok: false, models: [] };
    if (provider.wire === 'gemini' && Array.isArray(body.models)) {
      for (const raw of body.models as unknown[]) {
        if (!isRecord(raw) || typeof raw.name !== 'string') continue;
        const id = raw.name.startsWith('models/') ? raw.name.slice('models/'.length) : raw.name;
        const context = typeof raw.context_length === 'number' ? raw.context_length : typeof raw.inputTokenLimit === 'number' ? raw.inputTokenLimit : null;
        models.push({ id, free: raw.free === true, context_length: context });
      }
    } else if (Array.isArray(body.data)) {
      for (const raw of body.data as unknown[]) {
        if (!isRecord(raw) || typeof raw.id !== 'string') continue;
        const context = typeof raw.context_length === 'number' ? raw.context_length : null;
        models.push({ id: raw.id, free: raw.free === true, context_length: context });
      }
    }
    return { provider, ok: true, models };
  } catch (err) {
    log(`probe ${provider.id} failed: ${err instanceof Error ? err.message : 'error'}`);
    return { provider, ok: false, models: [] };
  }
}

/** The one catalog: the register body's `catalog` and `--list`'s output, the same bytes. */
function buildCatalog(brief: Brief, probes: Map<string, ProbedProvider>): Catalog {
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const probe = probes.get(provider.id);
    const models = probe !== undefined && probe.ok ? probe.models : fallbackModels(brief, provider.id);
    providers.push({ id: provider.id, base_url: provider.path, key_env: provider.key_env, wire: provider.wire, models });
  }
  return { providers };
}

/** Without a probe (no env), the quality table's rows stand in for a provider's listing. */
function fallbackModels(brief: Brief, providerId: string): CatalogModel[] {
  const models: CatalogModel[] = [];
  for (const row of brief.ratings) {
    if (row.provider === providerId) models.push({ id: row.model, free: row.free === true, context_length: row.context_length });
  }
  return models;
}

function serializeCatalog(catalog: Catalog): string {
  return JSON.stringify(catalog);
}

// ---------- routing ----------
interface Candidate { provider: ProviderDir; model: string; rating: number; context: number; caps: { tools: boolean; vision: boolean; json: boolean }; }

function meets(row: RatingRow, requires: ManifestReq['requires']): boolean {
  if (requires.tools === true && row.tools !== true) return false;
  if (requires.vision === true && row.vision !== true) return false;
  if (requires.json === true && row.json !== true) return false;
  // Absent or null min_context means "no floor": any context length qualifies.
  if (typeof requires.min_context === 'number' && row.context_length < requires.min_context) return false;
  return true;
}

function candidatesFor(brief: Brief, probes: Map<string, ProbedProvider>, exhausted: Set<string>, calls: Map<string, number>, request: ManifestReq, body: Json): Candidate[] {
  const out: Candidate[] = [];
  for (const row of brief.ratings) {
    if (row.free !== true) continue;
    if (!meets(row, request.requires)) continue;
    const probe = probes.get(row.provider);
    if (probe === undefined || !probe.ok) continue;
    const listed = probe.models.find((m) => m.id === row.model && m.free === true);
    if (listed === undefined) continue;
    const provider = probe.provider;
    if (exhausted.has(provider.id)) continue;
    // A JSON-mode request needs a wire that carries response_format: openai or gemini.
    if (request.api === 'openai' && jsonModeOf(body) && provider.wire === 'anthropic') continue;
    out.push({ provider, model: row.model, rating: row.rating_milli, context: row.context_length, caps: { tools: row.tools, vision: row.vision, json: row.json } });
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const ca = calls.get(a.provider.id) ?? 0;
    const cb = calls.get(b.provider.id) ?? 0;
    if (ca !== cb) return ca - cb;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return out;
}

function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

// ---------- cross-wire translation ----------
interface Parsed {
  systemText: string;
  userShaText: string;
  tools: string[];
  maxTokens: number | null;
  stops: string[];
  imageSha: string | null;
  jsonMode: boolean;
}

function textParts(content: unknown): string[] {
  if (typeof content === 'string') return [content];
  const parts: string[] = [];
  if (Array.isArray(content)) {
    for (const block of content as unknown[]) {
      if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
    }
  }
  return parts;
}

function openAIParse(body: Json): Parsed {
  const messages = Array.isArray(body.messages) ? (body.messages as unknown[]).filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  let imageSha: string | null = null;
  for (const message of messages) {
    if (message.role === 'system') {
      const text = textParts(message.content).join('\n');
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = textParts(message.content).join('\n');
      if (Array.isArray(message.content)) {
        for (const block of message.content as unknown[]) {
          if (isRecord(block) && block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') imageSha = block.image_url.url;
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools as unknown[]) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') tools.push(tool.function.name);
    }
  }
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  const stops = stringList(body.stop);
  return { systemText: systemParts.join('\n'), userShaText: userText, tools, maxTokens, stops, imageSha, jsonMode: jsonModeOf(body) };
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return (v as unknown[]).filter((x): x is string => typeof x === 'string');
}

function blocksText(content: unknown): string {
  return textParts(content).join('\n');
}

function openAIToAnthropic(body: Json, parsed: Parsed, model: string): Json {
  const messages: Json[] = [];
  const src = Array.isArray(body.messages) ? (body.messages as unknown[]).filter(isRecord) : [];
  for (const message of src) {
    if (message.role === 'system') continue;
    const content: Json[] = [];
    const parts = Array.isArray(message.content) ? message.content as unknown[] : null;
    if (parts !== null) {
      for (const block of parts) {
        if (isRecord(block) && block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const data = /^data:([^,]*);base64,(.*)$/s.exec(block.image_url.url);
          if (data !== null) content.push({ type: 'image', source: { type: 'base64', media_type: data[1] || 'image/png', data: data[2] } });
          else content.push({ type: 'image', source: { type: 'url', url: block.image_url.url } });
        } else if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') {
          content.push({ type: 'text', text: block.text });
        }
      }
    } else {
      content.push({ type: 'text', text: blocksText(message.content) });
    }
    messages.push({ role: message.role === 'assistant' ? 'assistant' : 'user', content });
  }
  const tools: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools as unknown[]) {
      if (isRecord(tool) && isRecord(tool.function)) {
        tools.push({ name: tool.function.name, description: tool.function.description ?? '', input_schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} } });
      }
    }
  }
  const out: Json = { model, max_tokens: parsed.maxTokens, messages };
  if (parsed.systemText !== '') out.system = parsed.systemText;
  if (tools.length > 0) out.tools = tools;
  if (parsed.stops.length > 0) out.stop_sequences = parsed.stops;
  return out;
}

function openAIToGemini(body: Json, parsed: Parsed, model: string): Json {
  const contents: Json[] = [];
  const src = Array.isArray(body.messages) ? (body.messages as unknown[]).filter(isRecord) : [];
  for (const message of src) {
    if (message.role === 'system') continue;
    const parts: Json[] = [];
    const partsIn = Array.isArray(message.content) ? message.content as unknown[] : null;
    if (partsIn !== null) {
      for (const block of partsIn) {
        if (isRecord(block) && block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const data = /^data:([^,]*);base64,(.*)$/s.exec(block.image_url.url);
          if (data !== null) parts.push({ inlineData: { mimeType: data[1] || 'image/png', data: data[2] } });
          else parts.push({ fileData: { fileUri: block.image_url.url } });
        } else if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') {
          parts.push({ text: block.text });
        }
      }
    } else {
      parts.push({ text: blocksText(message.content) });
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const tools: Json[] = [];
  if (Array.isArray(body.tools) && (body.tools as unknown[]).length > 0) {
    const declarations: Json[] = [];
    for (const tool of body.tools as unknown[]) {
      if (isRecord(tool) && isRecord(tool.function)) {
        declarations.push({ name: tool.function.name, description: tool.function.description ?? '', parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} } });
      }
    }
    if (declarations.length > 0) tools.push({ functionDeclarations: declarations });
  }
  const config: Json = {};
  if (parsed.maxTokens !== null) config.maxOutputTokens = parsed.maxTokens;
  if (parsed.stops.length > 0) config.stopSequences = parsed.stops;
  if (parsed.jsonMode) config.responseMimeType = 'application/json';
  const out: Json = { contents };
  if (parsed.systemText !== '') out.systemInstruction = { parts: [{ text: parsed.systemText }] };
  if (tools.length > 0) out.tools = tools;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemText = blocksText(body.system);
  if (systemText !== '') messages.push({ role: 'system', content: systemText });
  const src = Array.isArray(body.messages) ? (body.messages as unknown[]).filter(isRecord) : [];
  for (const message of src) {
    const content: Json[] = [];
    const parts = Array.isArray(message.content) ? message.content as unknown[] : null;
    if (parts !== null) {
      for (const block of parts) {
        if (isRecord(block) && block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            content.push({ type: 'image_url', image_url: { url: `data:${typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png'};base64,${block.source.data}` } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            content.push({ type: 'image_url', image_url: { url: block.source.url } });
          }
        } else if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') {
          content.push({ type: 'text', text: block.text });
        }
      }
    } else {
      content.push({ type: 'text', text: blocksText(message.content) });
    }
    messages.push({ role: message.role === 'assistant' ? 'assistant' : 'user', content });
  }
  const tools: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools as unknown[]) {
      if (isRecord(tool) && typeof tool.name === 'string') {
        tools.push({ type: 'function', function: { name: tool.name, description: typeof tool.description === 'string' ? tool.description : '', parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} } } });
      }
    }
  }
  const out: Json = { model, messages, stream: false };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (tools.length > 0) out.tools = tools;
  if (Array.isArray(body.stop_sequences) && (body.stop_sequences as unknown[]).length > 0) out.stop = body.stop_sequences;
  return out;
}

function anthropicToGemini(body: Json, model: string): Json {
  const systemText = blocksText(body.system);
  const contents: Json[] = [];
  const src = Array.isArray(body.messages) ? (body.messages as unknown[]).filter(isRecord) : [];
  for (const message of src) {
    const parts: Json[] = [];
    const partsIn = Array.isArray(message.content) ? message.content as unknown[] : null;
    if (partsIn !== null) {
      for (const block of partsIn) {
        if (isRecord(block) && block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        } else if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') {
          parts.push({ text: block.text });
        }
      }
    } else {
      parts.push({ text: blocksText(message.content) });
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const tools: Json[] = [];
  if (Array.isArray(body.tools) && (body.tools as unknown[]).length > 0) {
    const declarations: Json[] = [];
    for (const tool of body.tools as unknown[]) {
      if (isRecord(tool) && typeof tool.name === 'string') {
        declarations.push({ name: tool.name, description: typeof tool.description === 'string' ? tool.description : '', parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} } });
      }
    }
    if (declarations.length > 0) tools.push({ functionDeclarations: declarations });
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences) && (body.stop_sequences as unknown[]).length > 0) config.stopSequences = body.stop_sequences;
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) out.tools = tools;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------- provider calls ----------
interface ProviderAnswer { ok: boolean; status: number; text: string; usage: { input: number; output: number } | null; stop: string | null; permanent: boolean; }

function deepFindMessage(value: unknown, needles: string[]): string | null {
  if (typeof value === 'string') {
    for (const needle of needles) if (value.includes(needle)) return value;
    return null;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      const found = deepFindMessage(item, needles);
      if (found !== null) return found;
    }
    return null;
  }
  if (isRecord(value)) {
    for (const key of Object.keys(value)) {
      const found = deepFindMessage(value[key], needles);
      if (found !== null) return found;
    }
  }
  return null;
}

async function callProvider(env: string, provider: ProviderDir, wire: Wire, path: string, payload: Json, urlPath: string): Promise<{ status: number; json: Json | null }> {
  const url = `${env}${path}${urlPath}`;
  const headers: Json = { ...authHeaders(provider), 'content-type': 'application/json' };
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  const response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  const text = await response.text();
  let json: Json | null = null;
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed)) json = parsed;
  } catch { /* non-JSON */ }
  return { status: response.status, json };
}

const PERMANENT_MARKS = ['budget is spent', 'is out'];
const TEMPORARY_MARK = 'refused this call';

async function providerInfer(brief: Brief, probes: Map<string, ProbedProvider>, exhausted: Set<string>, calls: Map<string, number>, request: ManifestReq, body: Json): Promise<ProviderAnswer> {
  const candidates = candidatesFor(brief, probes, exhausted, calls, request, body);
  if (candidates.length === 0) return { ok: false, status: 0, text: '', usage: null, stop: null, permanent: false };
  let firstFailure: ProviderAnswer | null = null;
  for (const candidate of candidates) {
    const provider = candidate.provider;
    const wire = provider.wire;
    let payload: Json;
    let urlPath: string;
    if (wire === request.api) {
      payload = { ...body, model: candidate.model, stream: false };
      urlPath = wire === 'openai' ? '/v1/chat/completions' : '/v1/messages';
    } else if (request.api === 'openai' && wire === 'anthropic') {
      payload = openAIToAnthropic(body, openAIParse(body), candidate.model);
      urlPath = '/v1/messages';
    } else if (request.api === 'openai' && wire === 'gemini') {
      payload = openAIToGemini(body, openAIParse(body), candidate.model);
      urlPath = `/v1beta/models/${encodeURIComponent(candidate.model)}:generateContent`;
    } else if (request.api === 'anthropic' && wire === 'openai') {
      payload = anthropicToOpenAI(body, candidate.model);
      urlPath = '/v1/chat/completions';
    } else {
      payload = anthropicToGemini(body, candidate.model);
      urlPath = `/v1beta/models/${encodeURIComponent(candidate.model)}:generateContent`;
    }
    calls.set(provider.id, (calls.get(provider.id) ?? 0) + 1);
    let result: { status: number; json: Json | null };
    try {
      result = await callProvider(ENV_URL, provider, wire, provider.path, payload, urlPath);
    } catch (err) {
      log(`call ${provider.id}/${candidate.model} threw: ${err instanceof Error ? err.message : 'error'}`);
      continue;
    }
    if (result.status >= 200 && result.status < 300 && result.json !== null) {
      const answer = extractAnswer(wire, result.json);
      if (answer !== null) return { ok: true, status: result.status, ...answer, permanent: false };
      continue;
    }
    const refusal: ProviderAnswer = { ok: false, status: result.status, text: '', usage: null, stop: null, permanent: false };
    if (result.json !== null) {
      const message = deepFindMessage(result.json, PERMANENT_MARKS);
      if (message !== null) refusal.permanent = true;
      else if (deepFindMessage(result.json, [TEMPORARY_MARK]) !== null) refusal.permanent = false;
      else refusal.permanent = result.status === 402;
    }
    if (refusal.permanent) exhausted.add(provider.id);
    if (firstFailure === null) firstFailure = refusal;
    log(`refused by ${provider.id}/${candidate.model}: ${result.status}${refusal.permanent ? ' (permanent)' : ''}`);
  }
  return firstFailure ?? { ok: false, status: 0, text: '', usage: null, stop: null, permanent: false };
}

function extractAnswer(wire: Wire, json: Json): { text: string; usage: { input: number; output: number } | null; stop: string | null } | null {
  if (wire === 'openai') {
    const choices = Array.isArray(json.choices) ? (json.choices as unknown[]) : null;
    if (choices === null || choices.length === 0 || !isRecord(choices[0])) return null;
    const choice = choices[0];
    const message = isRecord(choice.message) ? choice.message : null;
    if (message === null) return null;
    const text = typeof message.content === 'string' ? message.content : textParts(message.content).join('');
    const usage = isRecord(json.usage) && typeof json.usage.prompt_tokens === 'number' && typeof json.usage.completion_tokens === 'number'
      ? { input: json.usage.prompt_tokens, output: json.usage.completion_tokens } : null;
    const stop = typeof choice.finish_reason === 'string' ? choice.finish_reason : null;
    return { text, usage, stop };
  }
  if (wire === 'anthropic') {
    if (!Array.isArray(json.content)) return null;
    let text = '';
    for (const block of json.content as unknown[]) {
      if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') text += block.text;
    }
    const usage = isRecord(json.usage) && typeof json.usage.input_tokens === 'number' && typeof json.usage.output_tokens === 'number'
      ? { input: json.usage.input_tokens, output: json.usage.output_tokens } : null;
    const stop = typeof json.stop_reason === 'string' ? json.stop_reason : null;
    return { text, usage, stop };
  }
  const candidates = Array.isArray(json.candidates) ? (json.candidates as unknown[]) : null;
  if (candidates === null || candidates.length === 0 || !isRecord(candidates[0])) return null;
  const content = isRecord((candidates[0] as Json).content) ? (candidates[0] as Json).content as Json : null;
  let text = '';
  if (content !== null && Array.isArray(content.parts)) {
    const parts: string[] = [];
    for (const part of content.parts as unknown[]) {
      if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
    }
    text = parts.join('\n');
  }
  const usage = isRecord(json.usageMetadata) && typeof json.usageMetadata.promptTokenCount === 'number' && typeof json.usageMetadata.candidatesTokenCount === 'number'
    ? { input: json.usageMetadata.promptTokenCount, output: json.usageMetadata.candidatesTokenCount } : null;
  const candidate = candidates[0];
  const stop = typeof (candidate as Json).finishReason === 'string' ? (candidate as Json).finishReason as string : null;
  return { text, usage, stop };
}

// ---------- answering the client ----------
const STOP_TO_OPENAI: Map<string, string> = new Map([
  ['stop', 'stop'], ['end_turn', 'stop'], ['STOP', 'stop'], ['STOP_SEQUENCES', 'stop'],
  ['length', 'length'], ['max_tokens', 'length'], ['MAX_TOKENS', 'length'],
  ['tool_calls', 'tool_calls'], ['function_call', 'tool_calls'], ['tool_use', 'tool_calls'],
  ['content_filter', 'content_filter'], ['refusal', 'content_filter'], ['SAFETY', 'content_filter'],
]);
const STOP_TO_ANTHROPIC: Map<string, string> = new Map([
  ['stop', 'end_turn'], ['end_turn', 'end_turn'], ['STOP', 'end_turn'], ['STOP_SEQUENCES', 'end_turn'],
  ['length', 'max_tokens'], ['max_tokens', 'max_tokens'], ['MAX_TOKENS', 'max_tokens'],
  ['tool_calls', 'tool_use'], ['function_call', 'tool_use'], ['tool_use', 'tool_use'],
  ['content_filter', 'refusal'], ['refusal', 'refusal'], ['SAFETY', 'refusal'],
]);

function splitChunks(text: string, size: number): string[] {
  if (text === '') return [''];
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces;
}

function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function openAIClientAnswer(text: string, usage: { input: number; output: number } | null, stop: string | null, stream: boolean, model: string): { status: number; contentType: string; body: string } {
  const finish = STOP_TO_OPENAI.get(stop ?? 'stop') ?? 'stop';
  const tokens = usage ?? { input: 1, output: 1 };
  const usageObject = { prompt_tokens: tokens.input, completion_tokens: tokens.output, total_tokens: tokens.input + tokens.output };
  if (!stream) {
    const completion = { id: 'chatcmpl-entry', object: 'chat.completion', created: FIXED_CREATED, model, choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: finish, logprobs: null }], usage: usageObject };
    return { status: 200, contentType: 'application/json', body: JSON.stringify(completion) };
  }
  const chunk = (delta: Json, finishReason: string | null): Json => ({ id: 'chatcmpl-entry', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finishReason }] });
  const frames: string[] = [sseFrame(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitChunks(text, 64)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  frames.push(sseFrame(null, chunk({}, finish)));
  frames.push(sseFrame(null, { id: 'chatcmpl-entry', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: usageObject }));
  frames.push('data: [DONE]\n\n');
  return { status: 200, contentType: 'text/event-stream; charset=utf-8', body: frames.join('') };
}

function anthropicClientAnswer(text: string, usage: { input: number; output: number } | null, stop: string | null, stream: boolean, model: string): { status: number; contentType: string; body: string } {
  const reason = STOP_TO_ANTHROPIC.get(stop ?? 'end_turn') ?? 'end_turn';
  const tokens = usage ?? { input: 1, output: 1 };
  if (!stream) {
    const message = { id: 'msg_entry', type: 'message', role: 'assistant', model, content: [{ type: 'text', text }], stop_reason: reason, stop_sequence: null, usage: { input_tokens: tokens.input, output_tokens: tokens.output } };
    return { status: 200, contentType: 'application/json', body: JSON.stringify(message) };
  }
  const frames: string[] = [
    sseFrame('message_start', { type: 'message_start', message: { id: 'msg_entry', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: tokens.input, output_tokens: 0 } } }),
    sseFrame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitChunks(text, 64)) frames.push(sseFrame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  frames.push(
    sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseFrame('message_delta', { type: 'message_delta', delta: { stop_reason: reason, stop_sequence: null }, usage: { output_tokens: tokens.output } }),
    sseFrame('message_stop', { type: 'message_stop' }),
  );
  return { status: 200, contentType: 'text/event-stream; charset=utf-8', body: frames.join('') };
}

function errorBody(api: 'openai' | 'anthropic', status: number, message: string): { status: number; contentType: string; body: string } {
  if (api === 'openai') {
    const type = status === 429 ? 'rate_limit_error' : status >= 500 ? 'api_error' : 'invalid_request_error';
    return { status, contentType: 'application/json', body: JSON.stringify({ error: { message, type, param: null, code: null } }) };
  }
  const type = status === 401 ? 'authentication_error' : status === 400 ? 'invalid_request_error' : status === 429 ? 'rate_limit_error' : 'api_error';
  return { status, contentType: 'application/json', body: JSON.stringify({ type: 'error', error: { type, message } }) };
}

function freeModelList(catalog: Catalog): CatalogModel[] {
  const out: CatalogModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) if (model.free) out.push(model);
  }
  return out;
}

// ---------- the server ----------
const brief = readBrief(PROMPT_FILE);
const probes = new Map<string, ProbedProvider>();
let catalog: Catalog = { providers: [] };
const catalogBytes = (): string => serializeCatalog(catalog);
const exhausted = new Set<string>();
const calls = new Map<string, number>();
let answered = 0;

function clientApiOf(headers: IncomingMessage['headers']): 'openai' | 'anthropic' {
  return typeof headers['x-api-key'] === 'string' ? 'anthropic' : 'openai';
}

function readBody(request: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    request.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > 32 * 1024 * 1024) { reject(new Error('body too large')); request.destroy(); return; }
      chunks.push(chunk);
    });
    request.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    request.on('error', reject);
  });
}

async function handleRequest(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const api = clientApiOf(request.headers);
  const url = request.url ?? '/';
  if (request.method === 'GET' && url === '/v1/models') {
    const data = freeModelList(catalog).map((model) => ({ id: model.id, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator', free: true, context_length: model.context_length }));
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify(api === 'openai' ? { object: 'list', data } : { data: data.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: '2026-01-01T00:00:00Z', free: true, context_length: m.context_length })), has_more: false }));
    return;
  }
  if (request.method !== 'POST' || (url !== '/v1/chat/completions' && url !== '/v1/messages')) {
    const body = errorBody(api, 404, `no route ${request.method} ${url}`);
    response.writeHead(body.status, { 'content-type': body.contentType });
    response.end(body.body);
    return;
  }
  const raw = await readBody(request);
  let body: Json;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) throw new Error('not an object');
    body = parsed;
  } catch {
    const reply = errorBody(api, 400, 'the request body is not one JSON object');
    response.writeHead(reply.status, { 'content-type': reply.contentType });
    response.end(reply.body);
    return;
  }
  const stream = body.stream === true;
  const manifest = brief !== null ? brief.manifest[answered] ?? null : null;
  const requirements: ManifestReq['requires'] = manifest !== null ? manifest.requires : {};
  const requestMeta: ManifestReq = { id: manifest?.id ?? 'r', api, stream, weight: manifest?.weight ?? 1, requires: requirements };
  const answer = brief === null ? { ok: false, status: 503, text: '', usage: null, stop: null, permanent: false } : await providerInfer(brief, probes, exhausted, calls, requestMeta, body);
  let reply: { status: number; contentType: string; body: string };
  if (answer.ok) {
    answered += 1;
    const model = 'auto';
    reply = api === 'openai'
      ? openAIClientAnswer(answer.text, answer.usage, answer.stop, stream, model)
      : anthropicClientAnswer(answer.text, answer.usage, answer.stop, stream, model);
  } else {
    reply = errorBody(api, answer.status === 0 ? 502 : answer.status, 'every free model that meets this request refused');
  }
  response.writeHead(reply.status, { 'content-type': reply.contentType });
  response.end(reply.body);
}

function pickPort(): number {
  let hash = 2166136261;
  for (const byte of Buffer.from(SEED === '' ? 'no-seed' : SEED, 'utf8')) {
    hash ^= byte;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return 21000 + (hash % 8000);
}

async function main(): Promise<void> {
  if (process.argv.includes('--list')) {
    if (brief === null) { process.stdout.write('{"providers":[]}\n'); return; }
    if (ENV_URL !== '') {
      const results = await Promise.all(brief.providers.map((provider) => probeProvider(provider)));
      for (const probe of results) probes.set(probe.provider.id, probe);
    }
    const listCatalog = buildCatalog(brief, probes);
    process.stdout.write(`${serializeCatalog(listCatalog)}\n`);
    return;
  }
  if (brief === null || ENV_URL === '') {
    log('no brief or no env: nothing to do');
    process.exit(0);
  }
  const probeResults = await Promise.all(brief.providers.map((provider) => probeProvider(provider)));
  for (const probe of probeResults) probes.set(probe.provider.id, probe);
  catalog = buildCatalog(brief, probes);
  const port = pickPort();
  const server = createServer((request, response) => { void handleRequest(request, response); });
  await new Promise<void>((resolve) => server.listen(port, '0.0.0.0', resolve));
  log(`listening on ${port}`);
  let registered = false;
  let workPollFailures = 0;
  for (let attempt = 0; attempt < REGISTER_RETRIES && !registered; attempt++) {
    try {
      const response = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      const reply: unknown = await response.json();
      if (response.ok && isRecord(reply) && reply.ok === true) {
        registered = true;
        log(`registered: ${(reply.requests as number) ?? '?'} requests ahead`);
      }
    } catch (err) {
      log(`register attempt ${attempt + 1} failed: ${err instanceof Error ? err.message : 'error'}`);
    }
    if (!registered) await new Promise((resolve) => setTimeout(resolve, 200));
  }
  if (!registered) {
    log('could not register: exiting');
    process.exit(1);
  }
  for (;;) {
    try {
      const response = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(5000) });
      const reply: unknown = await response.json();
      if (isRecord(reply) && reply.done === true) break;
      workPollFailures = 0;
    } catch {
      // keep polling; surface a stall instead of spinning silently
      workPollFailures += 1;
      if (workPollFailures === 600) log('warn: /work unreachable for ~60 s, still polling');
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
  log(`done: ${answered} requests served`);
  server.close();
  process.exit(0);
}

void main();
