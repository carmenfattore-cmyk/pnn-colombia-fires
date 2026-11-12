// =====================================================================================
// Fire inside a protected area vs its 10 km external ring, Sentinel-2 (2017-2025)
// Google Earth Engine Code Editor - simplified, single-park version of the workflow of
// Fattore & Coronado-Hernandez (in prep., Journal of Environmental Management).
//
// What it does
//   1. Park polygon from WDPA; 10 km external ring (other protected areas removed).
//   2. Random sample points inside the park and in the ring (permanent water excluded).
//   3. For every year: Sentinel-2 L1C + Cloud Score+ mask, water masked with MNDWI;
//      per pixel NBR median, NBR minimum, date of the minimum, dNBR = median - minimum,
//      number of clear images and of "low" images (NBR < median - 0.1).
//   4. Fire confirmation: dNBR >= 0.15 in >= 2 clear images AND
//      (active-fire hotspot <= 500 m between -60 and +5 days of the minimum OR MODIS MCD64A1 burned).
//      Point-years with < 5 clear images are treated as missing (s2fire = -1).
//   5. Annual share of points burned, park vs ring, Sentinel-2 confirmed vs MODIS (chart),
//      export of the point table and of the annual summary to Google Drive.
//
// Notes
//   - Hotspots: by default the GEE 'FIRMS' collection (MODIS, 1 km). The article used the
//     VIIRS S-NPP 375 m archive downloaded from NASA FIRMS: upload that CSV as a table asset
//     (columns latitude, longitude, acq_date) and put its path in VIIRS_ASSET.
//   - In the article the extraction ran with the Python API in batches of points; matching,
//     GAM and LightGBM-SHAP were run in Python on the exported table.
//   - Interactive printing may time out for large parks: rely on the Tasks tab (exports).
// =====================================================================================

// ---------- 1. Settings ----------
var PARK_NAME = 'Isla de Salamanca';  // WDPA field NAME (check the printed list if nothing is found)
var ISO3 = 'COL';
var RING_KM = 10;
var YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
var N_POINTS = 1000;                  // points per zone (park, ring)
var CS_MIN = 0.6;                     // Cloud Score+ clear threshold
var DNBR_MIN = 0.15;                  // minimum NBR drop (median - minimum)
var DROP = 0.1;                       // an image is "low" if NBR < median - DROP
var MIN_LOW = 2;                      // at least MIN_LOW low images
var MIN_OBS = 5;                      // fewer clear images -> point-year missing
var HOT_DIST = 500;                   // m, hotspot search radius
var HOT_BEFORE = 60, HOT_AFTER = 5;   // days: hotspot within [min-60, min+5]
var VIIRS_ASSET = '';                 // e.g. 'projects/your-project/assets/fire_archive_SV-C2'
var SHOW_YEAR = 2024;                 // year displayed on the map
var SCALE = 20;
var SEED = 42;
var DRIVE_FOLDER = 'GEE_park_fire';

// ---------- 2. Park, ring and sample points ----------
var wdpa = ee.FeatureCollection('WCMC/WDPA/current/polygons')
  .filter(ee.Filter.eq('ISO3', ISO3))
  .filter(ee.Filter.neq('STATUS', 'Proposed'));
var parkFc = wdpa.filter(ee.Filter.eq('NAME', PARK_NAME));
print('WDPA polygons found for ' + PARK_NAME + ':', parkFc.size());
print('Names of protected areas in ' + ISO3 + ' (to check the spelling):',
      wdpa.aggregate_array('NAME').distinct().sort());

var park = parkFc.geometry().dissolve(100);
var outer = park.buffer(RING_KM * 1000, 100);
var others = wdpa.filterBounds(outer).filter(ee.Filter.neq('NAME', PARK_NAME)).geometry().dissolve(100);
var ring = outer.difference(park, 100).difference(others, 100);
var aoi = outer;

var wc = ee.ImageCollection('ESA/WorldCover/v200').first().rename('lc');

function samplePoints(geom, zone, seed) {
  var pts = ee.FeatureCollection.randomPoints({
    region: geom, points: Math.round(N_POINTS * 1.4), seed: seed, maxError: 50});
  return wc.sampleRegions({collection: pts, scale: 10, geometries: true})  // sea (masked) dropped
    .filter(ee.Filter.neq('lc', 80))                                        // permanent water dropped
    .limit(N_POINTS)
    .map(function (f) { return f.set('zone', zone); });
}
var points = samplePoints(park, 'park', SEED).merge(samplePoints(ring, 'ring', SEED + 1));

// ---------- 3. Sentinel-2 annual NBR metrics ----------
var S2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED');
var CSP = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

function prep(img) {
  var b = img.select(['B3', 'B8', 'B11', 'B12']).divide(10000);
  var nbr = b.normalizedDifference(['B8', 'B12']).rename('nbr');
  var mndwi = b.normalizedDifference(['B3', 'B11']);
  var doy = ee.Image.constant(
      ee.Date(img.get('system:time_start')).getRelative('day', 'year').add(1))
    .toInt16().rename('doy');
  var clear = img.select('cs').gte(CS_MIN).and(mndwi.lte(0));
  return nbr.addBands(doy).updateMask(clear);
}

function s2Metrics(y) {
  var col = S2.filterBounds(aoi)
    .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y + 1, 1, 1))
    .linkCollection(CSP, ['cs'])
    .map(prep);
  var med = col.select('nbr').median().rename('nbr_med');
  var mosaic = col.map(function (i) {
    return i.addBands(i.select('nbr').multiply(-1).rename('neg'));
  }).qualityMosaic('neg');
  var mn = mosaic.select('nbr').rename('nbr_min');
  var doyMin = mosaic.select('doy').rename('doy_min');
  var nlow = col.map(function (i) {
    return i.select('nbr').lt(med.subtract(DROP)).rename('nlow');
  }).sum();
  var nobs = col.select('nbr').count().rename('nobs');
  return med.addBands(mn).addBands(med.subtract(mn).rename('dnbr'))
    .addBands(doyMin).addBands(nlow).addBands(nobs);
}

// ---------- 4. MODIS burned area and hotspots ----------
function modisBurned(y) {
  return ee.ImageCollection('MODIS/061/MCD64A1')
    .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y + 1, 1, 1))
    .select('BurnDate').max().gt(0).unmask(0).rename('modis');
}

// hotspot image (GEE FIRMS collection): 1 where a hotspot lies within HOT_DIST and inside the
// time window around the date of the NBR minimum of that pixel
function hotspotImage(y, doyMin) {
  var jan1 = ee.Date.fromYMD(y, 1, 1);
  var col = ee.ImageCollection('FIRMS').select('T21')
    .filterBounds(aoi)
    .filterDate(jan1.advance(-HOT_BEFORE, 'day'), jan1.advance(1, 'year').advance(HOT_AFTER + 1, 'day'));
  var hits = col.map(function (img) {
    var d = ee.Date(img.get('system:time_start')).difference(jan1, 'day').add(1);
    var fire = img.gt(0).unmask(0)
      .focalMax({radius: HOT_DIST, units: 'meters', kernelType: 'circle'});
    var inWin = doyMin.gte(ee.Image.constant(d).subtract(HOT_AFTER))
      .and(doyMin.lte(ee.Image.constant(d).add(HOT_BEFORE)));
    return fire.and(inWin).rename('hot');
  });
  return ee.Image(ee.Algorithms.If(col.size().gt(0), hits.max(), ee.Image(0).rename('hot')))
    .unmask(0).rename('hot');
}

// hotspot check at the points from an uploaded VIIRS table (acq_date as 'YYYY-MM-DD')
function viirsAtPoints(fc, y) {
  var t = ee.FeatureCollection(VIIRS_ASSET).filterBounds(aoi);
  var jan1 = ee.Date.fromYMD(y, 1, 1);
  return fc.map(function (f) {
    var dmin = ee.Number(f.get('doy_min'));
    var start = jan1.advance(dmin.subtract(1 + HOT_BEFORE), 'day').format('YYYY-MM-dd');
    var end = jan1.advance(dmin.subtract(1).add(HOT_AFTER), 'day').format('YYYY-MM-dd');
    var n = t.filterBounds(f.geometry().buffer(HOT_DIST))
      .filter(ee.Filter.greaterThanOrEquals('acq_date', start))
      .filter(ee.Filter.lessThanOrEquals('acq_date', end)).size();
    return f.set('hot', ee.Algorithms.If(dmin.gt(0), n.gt(0), 0));
  });
}

// ---------- 5. Point table per year ----------
function yearTable(y) {
  var m = s2Metrics(y);
  var img = m.addBands(modisBurned(y));
  if (!VIIRS_ASSET) {
    img = img.addBands(hotspotImage(y, m.select('doy_min')));
  }
  var s = img.unmask(-999).sampleRegions({
    collection: points, scale: SCALE, geometries: true, tileScale: 4});
  if (VIIRS_ASSET) {
    s = viirsAtPoints(s, y);
  }
  return s.map(function (f) {
    var ok = ee.Number(f.get('nobs')).gte(MIN_OBS);
    var drop = ee.Number(f.get('dnbr')).gte(DNBR_MIN).and(ee.Number(f.get('nlow')).gte(MIN_LOW));
    var conf = drop.and(ee.Number(f.get('hot')).eq(1).or(ee.Number(f.get('modis')).eq(1)));
    return f.set({year: y, s2fire: ee.Algorithms.If(ok, conf, -1)});
  });
}

var table = ee.FeatureCollection(YEARS.map(yearTable)).flatten();

// ---------- 6. Annual summary: park vs ring ----------
function pct(fc, prop) {
  return ee.Number(fc.aggregate_mean(prop)).multiply(100);
}
var summary = ee.FeatureCollection(YEARS.map(function (y) {
  var t = table.filter(ee.Filter.eq('year', y)).filter(ee.Filter.gte('s2fire', 0));
  var p = t.filter(ee.Filter.eq('zone', 'park'));
  var r = t.filter(ee.Filter.eq('zone', 'ring'));
  var fires = t.filter(ee.Filter.eq('s2fire', 1));
  return ee.Feature(null, {
    year: y,
    park_s2: pct(p, 's2fire'), ring_s2: pct(r, 's2fire'),
    park_modis: pct(p, 'modis'), ring_modis: pct(r, 'modis'),
    s2_fires: fires.size(),
    missed_by_modis_pct: ee.Algorithms.If(fires.size().gt(0),
      pct(fires.filter(ee.Filter.eq('modis', 0)), 's2fire'), null)
  });
}));

print('Annual summary (% of points burned)', summary);
print(ui.Chart.feature.byFeature(summary, 'year',
    ['park_s2', 'ring_s2', 'park_modis', 'ring_modis'])
  .setOptions({
    title: PARK_NAME + ': share of points burned per year',
    vAxis: {title: '% of points'}, hAxis: {format: '####'},
    series: {0: {color: '#1b7837'}, 1: {color: '#1b7837', lineDashStyle: [4, 4]},
             2: {color: '#777777'}, 3: {color: '#777777', lineDashStyle: [4, 4]}},
    pointSize: 4}));

// ---------- 7. Exports ----------
Export.table.toDrive({collection: table, description: 'points_' + PARK_NAME.replace(/ /g, '_'),
  folder: DRIVE_FOLDER, fileFormat: 'CSV',
  selectors: ['zone', 'year', 'lc', 'nbr_med', 'nbr_min', 'dnbr', 'doy_min', 'nlow', 'nobs',
              'hot', 'modis', 's2fire', '.geo']});
Export.table.toDrive({collection: summary, description: 'summary_' + PARK_NAME.replace(/ /g, '_'),
  folder: DRIVE_FOLDER, fileFormat: 'CSV'});

// ---------- 8. Map ----------
var mShow = s2Metrics(SHOW_YEAR);
var firePix = mShow.select('dnbr').gte(DNBR_MIN).and(mShow.select('nlow').gte(MIN_LOW))
  .and(mShow.select('nobs').gte(MIN_OBS))
  .and(modisBurned(SHOW_YEAR).or(hotspotImage(SHOW_YEAR, mShow.select('doy_min'))));
Map.centerObject(park, 10);
Map.addLayer(mShow.select('dnbr').clip(aoi),
  {min: 0, max: 0.5, palette: ['ffffff', 'fee08b', 'f46d43', 'a50026']}, 'dNBR ' + SHOW_YEAR, false);
Map.addLayer(firePix.selfMask().clip(aoi), {palette: ['d73027']}, 'Confirmed fire ' + SHOW_YEAR);
Map.addLayer(modisBurned(SHOW_YEAR).selfMask().clip(aoi), {palette: ['2166ac']},
  'MODIS burned ' + SHOW_YEAR, false);
Map.addLayer(ee.Image().paint(ring, 1, 1), {palette: ['555555']}, '10 km ring');
Map.addLayer(ee.Image().paint(park, 1, 2), {palette: ['1b7837']}, 'Park');
Map.addLayer(points, {color: '000000'}, 'Sample points', false);
