from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("scoring_endpoint")

# Define the input schema using Pydantic
class CarModel(BaseModel):
    manufacture_date: int
    seats: float
    mileage_v2: float
    car_age: float
    brand_Chevrolet: bool
    brand_Ford: bool
    brand_Honda: bool
    brand_Hyundai: bool
    brand_Kia: bool
    brand_Mazda: bool
    brand_Mercedes_Benz: bool
    brand_Mitsubishi: bool
    brand_Suzuki: bool
    brand_Toyota: bool
    brand_Vinfast: bool
    model_1083: bool
    model_190: bool
    model_2: bool
    model_2_Series: bool
    model_3: bool
    model_3_Series: bool
    model_3000GT: bool
    model_323: bool
    model_4_Runner: bool
    model_4_Series: bool
    model_5: bool
    model_5_Series: bool
    model_6: bool
    model_6_Series: bool
    model_626: bool
    model_7_Series: bool
    model_86: bool
    model_929: bool
    model_A_Class: bool
    model_AMG: bool
    model_APV: bool
    model_Accent: bool
    model_Accord: bool
    model_Acononline: bool
    model_Aerio: bool
    model_Alphard: bool
    model_Alto: bool
    model_Aristo: bool
    model_Aspire: bool
    model_Astro: bool
    model_Atenza: bool
    model_Atos: bool
    model_Attrage: bool
    model_Aurion: bool
    model_Avalon: bool
    model_Avante: bool
    model_Avanza: bool
    model_Aveo: bool
    model_Aygo: bool
    model_Azera: bool
    model_B_Class: bool
    model_BR_V: bool
    model_BT_50: bool
    model_Balenno: bool
    model_Brio: bool
    model_C_Class: bool
    model_CD5: bool
    model_CL_Class: bool
    model_CLA_Class: bool
    model_CLK_Class: bool
    model_CLS_Class: bool
    model_CR_V: bool
    model_CR_X: bool
    model_CX_3: bool
    model_CX_5: bool
    model_CX_8: bool
    model_CX_9: bool
    model_CX_30: bool
    model_Cadenza: bool
    model_Caldina: bool
    model_Camaro: bool
    model_Cami: bool
    model_Camry: bool
    model_Captiva: bool
    model_Caravan: bool
    model_Carens: bool
    model_Carnival: bool
    model_Cavalier: bool
    model_Celerio: bool
    model_Celica: bool
    model_Centennial: bool
    model_Century: bool
    model_Cerato: bool
    model_Cerato_Koup: bool
    model_Challenger: bool
    model_Chevyvan: bool
    model_Ciaz: bool
    model_City: bool
    model_Civic: bool
    model_Click: bool
    model_Colorado: bool
    model_Colt: bool
    model_Concord: bool
    model_Contour: bool
    model_Corolla: bool
    model_Corolla_Altis: bool
    model_Corolla_Cross: bool
    model_Corona: bool
    model_County: bool
    model_Coupe: bool
    model_Cressida: bool
    model_Creta: bool
    model_Crown: bool
    model_Cruze: bool
    model_Cultis_wagon: bool
    model_Diamante: bool
    model_Dong_khac: bool
    model_E_Class: bool
    model_Eclipse: bool
    model_EcoSport: bool
    model_Elantra: bool
    model_Enterprise: bool
    model_Eon: bool
    model_Ertiga: bool
    model_Escape: bool
    model_Escort: bool
    model_Everest: bool
    model_Explorer: bool
    model_F_150: bool
    model_F_350: bool
    model_Fadil: bool
    model_Fiesta: bool
    model_Fj_cruiser: bool
    model_Focus: bool
    model_Focus_C_Max: bool
    model_Forte: bool
    model_Fortuner: bool
    model_Fucion: bool
    model_G_Class: bool
    model_GL_Class: bool
    model_GLA_Class: bool
    model_GLB: bool
    model_GLC: bool
    model_GLC_Class: bool
    model_GLE_Class: bool
    model_GLK_Class: bool
    model_GLS_Class: bool
    model_GT_Coupe: bool
    model_Galant: bool
    model_Galloper: bool
    model_Genesis: bool
    model_Getz: bool
    model_Gold: bool
    model_Grand_Starex: bool
    model_Grand_i10: bool
    model_Grand_vitara: bool
    model_Grandeur: bool
    model_Grandis: bool
    model_Grunder: bool
    model_H_1: bool
    model_HR_V: bool
    model_Hiace: bool
    model_Highlander: bool
    model_Hilux: bool
    model_IQ: bool
    model_Innova: bool
    model_Innova_Cross: bool
    model_Ioniq_5: bool
    model_Jazz: bool
    model_Jolie: bool
    model_K3: bool
    model_K5: bool
    model_Ka: bool
    model_Kona: bool
    model_L300: bool
    model_Lacetti: bool
    model_Lancer: bool
    model_Land_Cruiser: bool
    model_Land_Cruiser_Prado: bool
    model_Lantra: bool
    model_Laser: bool
    model_Legend: bool
    model_Libero: bool
    model_Liteace: bool
    model_Lumina: bool
    model_Lux_A2_0: bool
    model_Lux_SA2_0: bool
    model_M_Class: bool
    model_M_couper: bool
    model_M3: bool
    model_M6: bool
    model_MB: bool
    model_ML_Class: bool
    model_MX_3: bool
    model_MX_6: bool
    model_Magentis: bool
    model_Matiz: bool
    model_Maybach: bool
    model_Mighty: bool
    model_Mirage: bool
    model_Mondeo: bool
    model_Morning: bool
    model_Mustang: bool
    model_NSX: bool
    model_Nubira: bool
    model_Odyssey: bool
    model_Optima: bool
    model_Orlando: bool
    model_Outlander: bool
    model_Outlander_Sport: bool
    model_Pajero: bool
    model_Pajero_Sport: bool
    model_Picanto: bool
    model_Pilot: bool
    model_Potentia: bool
    model_Prado: bool
    model_Pregio: bool
    model_Premacy: bool
    model_Previa: bool
    model_Pride: bool
    model_Prius: bool
    model_Quoris: bool
    model_R_Class: bool
    model_RAV4: bool
    model_Raize: bool
    model_Ranger: bool
    model_Rio: bool
    model_Rondo: bool
    model_Rush: bool
    model_S_10: bool
    model_S_Class: bool
    model_SLK_Class: bool
    model_SSR: bool
    model_Samirai: bool
    model_Santa_Fe: bool
    model_Sedona: bool
    model_Seltos: bool
    model_Sephia: bool
    model_Sequoia: bool
    model_Sienna: bool
    model_Solara: bool
    model_Solati: bool
    model_Soluto: bool
    model_Sonata: bool
    model_Sonet: bool
    model_Sorento: bool
    model_Sota: bool
    model_Soul: bool
    model_Spark: bool
    model_Spectra: bool
    model_Sportage: bool
    model_Sprinter: bool
    model_Starex: bool
    model_Stargazer: bool
    model_Swift: bool
    model_Tercel: bool
    model_Terracan: bool
    model_Territory: bool
    model_Tiburon: bool
    model_Tourneo: bool
    model_Trailblazer: bool
    model_Trajet: bool
    model_Transit: bool
    model_Trax: bool
    model_Tribute: bool
    model_Triton: bool
    model_Tucson: bool
    model_Tundra: bool
    model_Tuscani: bool
    model_Universe: bool
    model_Universe_Xpress_Luxury: bool
    model_V_Class: bool
    model_VF5: bool
    model_VF8: bool
    model_VF9: bool
    model_VFe34: bool
    model_Veloster: bool
    model_Veloz: bool
    model_Veloz_Cross: bool
    model_Venza: bool
    model_Veracruz: bool
    model_Verna: bool
    model_Vigor: bool
    model_Vios: bool
    model_Vista: bool
    model_Vitara: bool
    model_Vito: bool
    model_Vivant: bool
    model_Wagon_R_plus: bool
    model_Wigo: bool
    model_Wind_star: bool
    model_Wish: bool
    model_X1: bool
    model_X2: bool
    model_X3: bool
    model_X4: bool
    model_X5: bool
    model_X6: bool
    model_X7: bool
    model_XG: bool
    model_XL_7: bool
    model_Xpander: bool
    model_Xpander_Cross: bool
    model_Yaris: bool
    model_Yaris_Verso: bool
    model_Z4: bool
    model_Z4_Roadster: bool
    model_Zace: bool
    model_Zinger: bool
    model_eMighty: bool
    model_i20: bool
    model_i3: bool
    model_i30: bool
    model_i8: bool
    origin_My: bool
    origin_Nhat_Ban: bool
    origin_Nuoc_khac: bool
    origin_Thai_Lan: bool
    origin_Trung_Quoc: bool
    origin_Viet_Nam: bool
    origin_Dai_Loan: bool
    origin_Duc: bool
    origin_An_Do: bool
    type_hatchback: bool
    type_sedan: bool
    type_minivan_mpv: bool
    type_suv_cross_over: bool
    type_pickup: bool
    type_mui_tran: bool
    gearbox_AT: bool
    gearbox_MT: bool
    gearbox_UNKNOWN: bool
    fuel_petrol: bool
    fuel_hybrid: bool
    fuel_oil: bool
    color_blue: bool
    color_brown: bool
    color_gold: bool
    color_green: bool
    color_grey: bool
    color_orange: bool
    color_others: bool
    color_pink: bool
    color_red: bool
    color_silver: bool
    color_white: bool

# Load the trained model
model = joblib.load('./model/baseline_mlp.pkl')

@app.post("/")
async def scoring_endpoint(car: CarModel):
    try:
        # Step 1: Convert input into DataFrame
        input_data = pd.DataFrame([car.dict().values()], columns=car.dict().keys())
        logger.debug(f"Input Data Received:\n{input_data}")

        # Step 2: Ensure all values are in the correct format for the model
        input_data = input_data.astype(float)  # Convert all values to float
        logger.debug(f"Input Data After Conversion to Float:\n{input_data}")

        # Step 3: Make predictions
        try:
            yhat = model.predict(input_data)
        except Exception as e:
            logger.error(f"Error During Prediction: {e}")
            raise e

        # Log the prediction
        logger.debug(f"Prediction: {yhat}")

        # Step 4: Return the prediction as JSON
        return {"prediction": float(yhat[0])}

    except Exception as e:
        logger.error(f"Error in Scoring Endpoint: {e}")
        return {"error": str(e)}
