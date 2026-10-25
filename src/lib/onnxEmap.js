// Minimal ONNX protobuf reader: finds a [512, 512] float initializer (inswapper's
// "emap") inside the raw model bytes. ONNX Runtime Web has no getInitializer(),
// so we read it from the file ourselves.
//
// ModelProto.graph = field 7, GraphProto.initializer = field 5,
// TensorProto: dims = 1, data_type = 2, float_data = 4, name = 8, raw_data = 9.

function readVarint(bytes, pos) {
  let result = 0;
  let mul = 1;
  for (;;) {
    const b = bytes[pos++];
    result += (b & 0x7f) * mul;
    if (b < 0x80) return [result, pos];
    mul *= 128;
  }
}

// Calls visit(field, wireType, start, end) for every field in [start, end).
function walk(bytes, start, end, visit) {
  let pos = start;
  while (pos < end) {
    let key;
    [key, pos] = readVarint(bytes, pos);
    const field = Math.floor(key / 8);
    const wire = key & 7;
    if (wire === 0) {
      const s = pos;
      [, pos] = readVarint(bytes, pos);
      visit(field, wire, s, pos);
    } else if (wire === 1) {
      visit(field, wire, pos, pos + 8);
      pos += 8;
    } else if (wire === 2) {
      let len;
      [len, pos] = readVarint(bytes, pos);
      visit(field, wire, pos, pos + len);
      pos += len;
    } else if (wire === 5) {
      visit(field, wire, pos, pos + 4);
      pos += 4;
    } else {
      throw new Error(`Formato ONNX no soportado (wire type ${wire}).`);
    }
  }
}

function parseTensor(bytes, start, end) {
  const dims = [];
  let name = '';
  let dataType = 0;
  let raw = null;
  let floatRange = null;
  walk(bytes, start, end, (field, wire, s, e) => {
    if (field === 1) {
      if (wire === 0) dims.push(readVarint(bytes, s)[0]);
      else if (wire === 2) {
        let p = s;
        while (p < e) {
          let v;
          [v, p] = readVarint(bytes, p);
          dims.push(v);
        }
      }
    } else if (field === 2 && wire === 0) dataType = readVarint(bytes, s)[0];
    else if (field === 8 && wire === 2) name = new TextDecoder().decode(bytes.subarray(s, e));
    else if (field === 9 && wire === 2) raw = [s, e];
    else if (field === 4 && wire === 2) floatRange = [s, e];
  });
  return { dims, name, dataType, raw, floatRange };
}

/**
 * @param {Uint8Array} bytes the full .onnx file
 * @returns {Float32Array} row-major 512x512 matrix
 */
export function extractEmap(bytes) {
  let found = null;
  let fallback = null;
  walk(bytes, 0, bytes.length, (field, wire, s, e) => {
    if (field !== 7 || wire !== 2) return; // graph
    walk(bytes, s, e, (gField, gWire, gs, ge) => {
      if (gField !== 5 || gWire !== 2) return; // initializer
      const t = parseTensor(bytes, gs, ge);
      const is512 = t.dims.length === 2 && t.dims[0] === 512 && t.dims[1] === 512;
      if (!is512 || t.dataType !== 1) return; // 1 = FLOAT
      if (t.name === 'emap') found = t;
      else fallback = t; // insightface usa el último initializer
    });
  });
  const t = found ?? fallback;
  if (!t) throw new Error('No se encontró la matriz emap [512,512] en el modelo.');
  const range = t.raw ?? t.floatRange;
  if (!range) throw new Error('La matriz emap no tiene datos.');
  // slice() copia a un buffer alineado, necesario para Float32Array.
  const out = new Float32Array(bytes.slice(range[0], range[1]).buffer);
  if (out.length !== 512 * 512) throw new Error('La matriz emap tiene un tamaño inesperado.');
  return out;
}

/**
 * insightface: latent = normalize(emb) · emap, luego normalize(latent).
 * @param {Float32Array} embedding length 512
 * @param {Float32Array} emap row-major 512x512
 */
export function projectEmbedding(embedding, emap) {
  const n = 512;
  let norm = 0;
  for (let i = 0; i < n; i++) norm += embedding[i] * embedding[i];
  norm = Math.sqrt(norm) || 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = embedding[i] / norm;
    if (v === 0) continue;
    const row = i * n;
    for (let j = 0; j < n; j++) out[j] += v * emap[row + j];
  }
  let outNorm = 0;
  for (let j = 0; j < n; j++) outNorm += out[j] * out[j];
  outNorm = Math.sqrt(outNorm) || 1;
  for (let j = 0; j < n; j++) out[j] /= outNorm;
  return out;
}
