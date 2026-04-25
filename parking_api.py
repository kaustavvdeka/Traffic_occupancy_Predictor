import os
import time
import random
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
# Import the existing ParkingPredictor and new models
from parking_model import ParkingPredictor, ForecastingModel, DemandClusterer

app = Flask(__name__)
CORS(app)

model_path = 'parking_model.pkl'

# Initialize Models globally
predictor = None
forecaster = ForecastingModel()
clusterer = DemandClusterer()

try:
    if os.path.exists(model_path):
        predictor = ParkingPredictor()
        predictor.load_model(model_path)
        print("✅ Traffic Parking Model loaded successfully.")
    else:
        print("⚠️ Warning: parking_model.pkl not found! Running in simulation mode.")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

# Pre-defined parking sites in Shillong area representing the nearby locations
PARKING_SITES = [
    {
        "id": "park1",
        "name": "Police Bazar Multi-Level Parking",
        "address": "Police Bazar, Shillong",
        "capacity": 200,
        "base_occupancy_rate": 0.85,
        "price": 50,
        "currency": "INR",
        "distance": "0.8 km",
        "image_url": "https://images.unsplash.com/photo-1590674899484-mww3r14p?auto=format&fit=crop&q=80&w=800",
        "type": "multilevel",
        "latitude": 25.5788,
        "longitude": 91.8933
    },
    {
        "id": "park2",
        "name": "Polo Grounds Parking Area",
        "address": "Polo Grounds, Shillong",
        "capacity": 350,
        "base_occupancy_rate": 0.4,
        "price": 30,
        "currency": "INR",
        "distance": "2.1 km",
        "image_url": "https://images.unsplash.com/photo-1506521781263-d8422e82f27a?auto=format&fit=crop&q=80&w=800",
        "type": "surface",
        "latitude": 25.5843,
        "longitude": 91.8978
    },
    {
        "id": "park3",
        "name": "Ward's Lake Parking",
        "address": "Near Ward's Lake, Shillong",
        "capacity": 80,
        "base_occupancy_rate": 0.92,
        "price": 40,
        "currency": "INR",
        "distance": "1.2 km",
        "image_url": "https://images.unsplash.com/photo-1621245082121-72f3d2f95bcb?auto=format&fit=crop&q=80&w=800",
        "type": "surface",
        "latitude": 25.5746,
        "longitude": 91.8845
    },
    {
        "id": "park4",
        "name": "Laitumkhrah Market Parking",
        "address": "Laitumkhrah, Shillong",
        "capacity": 120,
        "base_occupancy_rate": 0.75,
        "price": 60,
        "currency": "INR",
        "distance": "3.5 km",
        "image_url": "https://images.unsplash.com/photo-1573348722427-f1d6819fdf98?auto=format&fit=crop&q=80&w=800",
        "type": "multilevel",
        "latitude": 25.5684,
        "longitude": 91.8992
    }
]

def _calculate_live_data():
    """Generates the live enriched data for all sites"""
    now = datetime.now()
    hour = now.hour
    day_of_week = now.weekday()
    results = []

    # First pass: calculate base occupancies
    for site in PARKING_SITES:
        if predictor is not None:
            fluctuation = random.uniform(-0.1, 0.1)
            predicted_occupancy_rate = min(1.0, max(0.0, site['base_occupancy_rate'] + fluctuation))
        else:
            fluctuation = random.uniform(-0.15, 0.15)
            predicted_occupancy_rate = min(1.0, max(0.0, site['base_occupancy_rate'] + fluctuation))

        occupancy_percent = int(predicted_occupancy_rate * 100)
        
        # XGBoost Time-Series Forecasting for +15m and +30m
        pred_15 = forecaster.predict(predicted_occupancy_rate, hour, day_of_week, 15)
        pred_30 = forecaster.predict(predicted_occupancy_rate, hour, day_of_week, 30)

        site_data = {
            "id": site['id'],
            "name": site['name'],
            "address": site['address'],
            "capacity": site['capacity'],
            "occupied_spots": int(site['capacity'] * predicted_occupancy_rate),
            "available_spots": site['capacity'] - int(site['capacity'] * predicted_occupancy_rate),
            "occupancy_percent": occupancy_percent,
            "predicted_15min_percent": int(pred_15 * 100),
            "predicted_30min_percent": int(pred_30 * 100),
            "price": site['price'],
            "currency": site['currency'],
            "distance": site['distance'],
            "image_url": site['image_url'],
            "type": site['type'],
            "latitude": site['latitude'],
            "longitude": site['longitude'],
            "is_full": occupancy_percent >= 98
        }
        results.append(site_data)
        
    # Second pass: Apply clustering (DBSCAN) based on live occupancy & geo
    clusters = clusterer.cluster_sites(results)
    
    # Calculate congestion per cluster to distribute to sites
    cluster_congestion = {}
    for cl_id in set(clusters.values()):
        cluster_sites = [s for s in results if clusters[s['id']] == cl_id]
        if not cluster_sites:
            continue
        avg_occ = sum(s['occupancy_percent'] for s in cluster_sites) / len(cluster_sites)
        if avg_occ > 85:
            level = "high"
        elif avg_occ > 60:
            level = "moderate"
        else:
            level = "low"
        cluster_congestion[cl_id] = level
        
    for r in results:
        cl_id = clusters[r['id']]
        r['cluster_id'] = f"ZONE-{cl_id+1}"
        r['congestion_level'] = cluster_congestion[cl_id]
        
    return results

@app.route('/api/parking/nearby', methods=['GET'])
def get_nearby_parking():
    results = _calculate_live_data()
    return jsonify({"success": True, "data": results})

@app.route('/api/parking/forecast/<string:site_id>', methods=['GET'])
def get_forecast(site_id):
    results = _calculate_live_data()
    site = next((s for s in results if s['id'] == site_id), None)
    if not site:
        return jsonify({"success": False, "error": "Site not found"}), 404
        
    return jsonify({
        "success": True,
        "data": {
            "id": site_id,
            "current": site['occupancy_percent'],
            "in_15": site['predicted_15min_percent'],
            "in_30": site['predicted_30min_percent']
        }
    })

@app.route('/api/parking/reserve', methods=['POST'])
def reserve_parking():
    req_data = request.get_json() or {}
    site_id = req_data.get('site_id')
    user_id = req_data.get('user_id', 'anonymous')
    
    if not site_id:
        return jsonify({"success": False, "error": "site_id required"}), 400
        
    # In a real system, we would decrement available slots and save to DB
    # For now, we simulate an optimistic success.
    # Expiry is set 15 minutes from now.
    expiry = datetime.now().timestamp() + (15 * 60)
    
    return jsonify({
        "success": True,
        "data": {
            "reservation_id": f"res_{int(time.time())}",
            "site_id": site_id,
            "user_id": user_id,
            "status": "held",
            "expires_at": expiry
        }
    })

@app.route('/api/parking/clusters', methods=['GET'])
def get_parking_clusters():
    results = _calculate_live_data()
    
    # Group by cluster ID
    clusters = {}
    for s in results:
        cid = s['cluster_id']
        if cid not in clusters:
            clusters[cid] = {
                "cluster_id": cid,
                "congestion_level": s['congestion_level'],
                "sites": []
            }
        clusters[cid]["sites"].append(s['id'])
        
    return jsonify({"success": True, "data": list(clusters.values())})

@app.route('/api/parking/predict_image', methods=['POST'])
def predict_parking_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
        
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    image_file = request.files['image']
    temp_path = f"temp_{int(time.time())}.jpg"
    image_file.save(temp_path)
    
    try:
        results = predictor.predict_from_image(temp_path)
        os.remove(temp_path)
        return jsonify({"success": True, "data": results})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
