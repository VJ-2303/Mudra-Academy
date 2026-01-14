/**
 * ================================================================================
 * MUDRA CONSTANTS & NORMALIZATION
 * ================================================================================
 * Single Source of Truth for Mudra Names across the application.
 * Handles normalization of ML model outputs, Rule names, and UI display names.
 * ================================================================================
 */

// 1. CANONICAL LIST (The correct UI display names)
const CANONICAL_MUDRAS = [
    'Pataka', 'Tripataka', 'Ardhapataka', 'Kartarimukha', 'Mayura',
    'Ardhachandra', 'Arala', 'Shukatunda', 'Mushti', 'Shikhara',
    'Kapittha', 'Katakamukha', 'Suchi', 'Chandrakala', 'Padmakosha',
    'Sarpashirsha', 'Mrigashirsha', 'Simhamukha', 'Kangula', 'Alapadma',
    'Chatura', 'Bhramara', 'Hamsasya', 'Hamsapaksha', 'Sandamsha',
    'Mukula', 'Tamrachuda', 'Trishula'
];

// 2. NORMALIZATION MAP (Typos/Variations -> Canonical Name)
const MUDRA_NORMALIZATION_MAP = {
    // Model Output / Common Typos      -> Canonical Name
    'pathaka': 'Pataka',
    'pataka mudra': 'Pataka',

    'tripathaka': 'Tripataka',
    'tripataka mudra': 'Tripataka',

    'ardhapathaka': 'Ardhapataka',
    'ardhapataka mudra': 'Ardhapataka',

    'katrimukha': 'Kartarimukha',
    'kartarimukham': 'Kartarimukha',
    'kartari mukham mudra': 'Kartarimukha',

    'mayura mudra': 'Mayura',

    'ardhachandran': 'Ardhachandra',
    'ardhachandra mudra': 'Ardhachandra',

    'aralam': 'Arala',
    'arala mudra': 'Arala',

    'sukatunda': 'Shukatunda',
    'shukatundam': 'Shukatunda',
    'shuka tundam mudra': 'Shukatunda',

    'musthi': 'Mushti',
    'musti': 'Mushti',
    'musthi mudra': 'Mushti',

    'shikharam': 'Shikhara',
    'sikharam': 'Shikhara',
    'shikharam mudra': 'Shikhara',

    'kapith': 'Kapittha',
    'kapittha mudra': 'Kapittha',

    'katakamukha mudra': 'Katakamukha',

    'suchi mudra': 'Suchi',

    'chandrakala mudra': 'Chandrakala',

    'padmakosa': 'Padmakosha',
    'padmakosha mudra': 'Padmakosha',

    'sarpasirsha': 'Sarpashirsha',
    'sarpashirsha mudra': 'Sarpashirsha',

    'mrigasirsha': 'Mrigashirsha',
    'mrugashirsha': 'Mrigashirsha',
    'mrigasheersha': 'Mrigashirsha',
    'mrigasheersha mudra': 'Mrigashirsha',

    'simhamukham': 'Simhamukha',
    'simhamukha mudra': 'Simhamukha',

    'kangulam': 'Kangula',
    'kangula mudra': 'Kangula',

    'alapadmam': 'Alapadma',
    'alapadma mudra': 'Alapadma',

    'chaturam': 'Chatura',
    'chautra': 'Chatura',
    'chatura mudra': 'Chatura',

    'bramaram': 'Bhramara',
    'bharma': 'Bhramara',
    'bhramara mudra': 'Bhramara',

    'hamsasyam': 'Hamsasya',
    'hamsasya mudra': 'Hamsasya',

    'hamsapaksha': 'Hamsapaksha',
    'hamsapaksa': 'Hamsapaksha',
    'hamsapaksha mudra': 'Hamsapaksha',

    'sandamsha mudra': 'Sandamsha',

    'mukulam': 'Mukula',
    'mukula mudra': 'Mukula',

    'tamarachudam': 'Tamrachuda',
    'tamracuda': 'Tamrachuda',
    'tamrachuda mudra': 'Tamrachuda',

    'trishulam': 'Trishula',
    'trisula': 'Trishula',
    'trishula mudra': 'Trishula',

    // Shanka (Conch - double-hand but may appear)
    'shanka': 'Shanka',
    'shanka mudra': 'Shanka',

    // Anjali / Samyukta (if needed later)
    'anjali mudra': 'Anjali'
};

// 3. ASSET MAP (Canonical Name -> Filename Base)
// Handles cases where asset filenames don't match canonical names
const MUDRA_ASSET_MAP = {
    'Chatura': 'Chautra',
    'Bhramara': 'Bharma',
    'Mrigashirsha': 'Mrugashirsha', // Asset often named Mrugashirsha-Hasta.jpg
    'Shukatunda': 'Sukatunda',
    'Mushti': 'Musti',
    'Shikhara': 'Sikharam',
    'Tamrachuda': 'Tamracuda',
    'Trishula': 'Trisula',
    'Sarpashirsha': 'Sarpashirsa',
    'Padmakosha': 'Padmakosa',
    'Hamsapaksha': 'Hamsapaksa'
};

/**
 * Normalizes a mudra name to its Canonical form.
 * @param {string} name - The input name (e.g. "Pathaka", "Musthi Mudra")
 * @returns {string} - The standardized name (e.g. "Pataka", "Mushti") or original if not found
 */
function normalizeMudraName(name) {
    if (!name) return "";

    const lower = name.toLowerCase().trim();

    // Check direct map
    if (MUDRA_NORMALIZATION_MAP[lower]) {
        return MUDRA_NORMALIZATION_MAP[lower];
    }

    // Common cleanup if not in map
    const clean = lower.replace(" mudra", "").replace(/\s+/g, "");

    // Check map again with cleaned name
    if (MUDRA_NORMALIZATION_MAP[clean]) {
        return MUDRA_NORMALIZATION_MAP[clean];
    }

    // Return original (capitalized) if no match found, or try to capitalize it
    return name.charAt(0).toUpperCase() + name.slice(1);
}

/**
 * Gets the asset filename base for a canonical mudra name.
 * @param {string} canonicalName 
 * @returns {string} Filename base (e.g., "Chautra" for "Chatura")
 */
function getMudraAssetBase(canonicalName) {
    return MUDRA_ASSET_MAP[canonicalName] || canonicalName;
}

// Export
window.MudraConstants = {
    CANONICAL_MUDRAS,
    MUDRA_NORMALIZATION_MAP,
    MUDRA_ASSET_MAP,
    normalizeMudraName,
    getMudraAssetBase
};
