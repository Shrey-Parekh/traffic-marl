from src.config import PEAK_HOUR_CONFIG, INJECTION_CONFIG

print("PEAK_HOUR_CONFIG morning_peak:")
print(f"  NS_multiplier: {PEAK_HOUR_CONFIG['morning_peak']['NS_multiplier']}")
print(f"  EW_multiplier: {PEAK_HOUR_CONFIG['morning_peak']['EW_multiplier']}")
print()
print("INJECTION_CONFIG:")
print(f"  base_rate: {INJECTION_CONFIG['base_rate']}")
