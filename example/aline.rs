//! The demo's ALINE config: 9 consonants and 3 vowels with Kondrak's standard
//! costs, saliences, and feature values. It uses the MangoCats variant, so stress
//! marks (ˈ) count.

use pho::algorithms::Aline;
use pho::algorithms::aline::config::enum_map::enum_map;
use pho::algorithms::aline::config::{
    Airstream, AlignmentMode, AlineVariant, Back, Binary, CommonFeatures, ConsonantFeatures, Costs,
    FeatureValues, High, Manner, Phonation, PhoneticFeatures, Place, Salience,
    SecondaryArticulation, StressScopeFallback, VowelFeatures,
};
use std::collections::HashMap;

/// Builds the demo ALINE config with the given saliences.
pub fn demo_aline(salience: Salience) -> Aline {
    let costs = Costs::try_new(
        -10.0, // skip
        35.0,  // substitute
        45.0,  // expand_compress
        5.0,   // vowel_consonant
    )
    .expect("demo costs are valid");
    let mut aline = Aline::try_new(costs, salience, values(), sounds(), 0.001)
        .expect("demo ALINE config is valid");
    aline.set_variant(AlineVariant::MangoCats);
    aline.set_alignment_mode(AlignmentMode::Global);
    aline.set_stress_scope_fallback(StressScopeFallback::NextVowel);
    aline
}

/// Kondrak's standard saliences, with `voice` and `stress` given explicitly.
pub fn salience(voice: f32, stress: f32) -> Salience {
    Salience::try_new(
        5.0,   // syllabic
        40.0,  // place
        50.0,  // manner
        voice, // voice
        20.0,  // nasal
        10.0,  // retroflex
        10.0,  // lateral
        5.0,   // aspirated
        0.0,   // long
        3.0,   // high
        2.0,   // back
        2.0,   // round
        10.0,  // phonation
        10.0,  // airstream
        10.0,  // secondary
        stress,
    )
    .expect("demo saliences are valid")
}

/// Kondrak's standard position of each feature value on its `[0, 1]` scale.
fn values() -> FeatureValues {
    FeatureValues::try_new(
        enum_map! {
            Place::Bilabial => 1.0,
            Place::Labiodental => 0.95,
            Place::Linguolabial => 0.92,
            Place::Dental => 0.9,
            Place::Alveolar => 0.85,
            Place::Retroflex => 0.8,
            Place::PalatoAlveolar => 0.75,
            Place::AlveoloPalatal => 0.72,
            Place::Palatal => 0.7,
            Place::Velar => 0.6,
            Place::Uvular => 0.5,
            Place::Pharyngeal => 0.3,
            Place::Epiglottal => 0.2,
            Place::Glottal => 0.1,
            Place::Labiovelar => 1.0,
            Place::Vowel => -1.0,
        },
        enum_map! {
            Manner::Stop => 1.0,
            Manner::Affricate => 0.9,
            Manner::Fricative => 0.85,
            Manner::LateralFricative => 0.8,
            Manner::Trill => 0.7,
            Manner::Tap => 0.65,
            Manner::LateralFlap => 0.6,
            Manner::Approximant => 0.6,
            Manner::HighVowel => 0.4,
            Manner::MidVowel => 0.2,
            Manner::LowVowel => 0.0,
            Manner::Vowel => 0.5,
        },
        enum_map! {
            High::High => 1.0,
            High::NearHigh => 0.85,
            High::HighMid => 0.65,
            High::Mid => 0.5,
            High::LowMid => 0.35,
            High::NearLow => 0.15,
            High::Low => 0.0,
        },
        enum_map! {
            Back::Front => 1.0,
            Back::NearFront => 0.75,
            Back::Central => 0.5,
            Back::NearBack => 0.25,
            Back::Back => 0.0,
        },
        enum_map! {
            Binary::Plus => 1.0,
            Binary::Minus => 0.0,
        },
        enum_map! {
            Phonation::Modal => 1.0,
            Phonation::Breathy => 0.75,
            Phonation::Creaky => 0.5,
            Phonation::Stiff => 0.25,
            Phonation::Slack => 0.0,
        },
        enum_map! {
            Airstream::Pulmonic => 1.0,
            Airstream::Ejective => 0.66,
            Airstream::Implosive => 0.33,
            Airstream::Click => 0.0,
        },
        enum_map! {
            SecondaryArticulation::None => 1.0,
            SecondaryArticulation::Labialized => 0.75,
            SecondaryArticulation::Palatalized => 0.5,
            SecondaryArticulation::Velarized => 0.25,
            SecondaryArticulation::Pharyngealized => 0.0,
        },
    )
    .expect("demo feature values are valid")
}

/// A pulmonic, modal, non-syllabic, unaspirated consonant.
fn consonant(place: Place, manner: Manner, voice: Binary, nasal: Binary) -> PhoneticFeatures {
    let common = CommonFeatures::new(
        place,
        manner,
        Binary::Minus, // syllabic
        voice,
        nasal,
        Binary::Minus, // retroflex
        Binary::Minus, // lateral
        Phonation::Modal,
        Airstream::Pulmonic,
        SecondaryArticulation::None,
    );
    PhoneticFeatures::Consonant(ConsonantFeatures::new(common, Binary::Minus))
}

/// A short, oral, voiced vowel.
fn vowel(manner: Manner, high: High, back: Back, round: Binary) -> PhoneticFeatures {
    let common = CommonFeatures::new(
        Place::Vowel,
        manner,
        Binary::Plus,  // syllabic
        Binary::Plus,  // voice
        Binary::Minus, // nasal
        Binary::Minus, // retroflex
        Binary::Minus, // lateral
        Phonation::Modal,
        Airstream::Pulmonic,
        SecondaryArticulation::None,
    );
    PhoneticFeatures::Vowel(VowelFeatures::new(common, back, high, Binary::Minus, round))
}

fn sounds() -> HashMap<String, PhoneticFeatures> {
    use Binary::{Minus, Plus};
    let stop = Manner::Stop;
    HashMap::from(
        [
            ("p", consonant(Place::Bilabial, stop, Minus, Minus)),
            ("b", consonant(Place::Bilabial, stop, Plus, Minus)),
            ("m", consonant(Place::Bilabial, stop, Plus, Plus)),
            ("t", consonant(Place::Alveolar, stop, Minus, Minus)),
            ("d", consonant(Place::Alveolar, stop, Plus, Minus)),
            ("n", consonant(Place::Alveolar, stop, Plus, Plus)),
            ("k", consonant(Place::Velar, stop, Minus, Minus)),
            ("g", consonant(Place::Velar, stop, Plus, Minus)),
            (
                "s",
                consonant(Place::Alveolar, Manner::Fricative, Minus, Minus),
            ),
            (
                "a",
                vowel(Manner::LowVowel, High::Low, Back::Central, Minus),
            ),
            (
                "i",
                vowel(Manner::HighVowel, High::High, Back::Front, Minus),
            ),
            ("u", vowel(Manner::HighVowel, High::High, Back::Back, Plus)),
        ]
        .map(|(ipa, f)| (ipa.to_string(), f)),
    )
}
