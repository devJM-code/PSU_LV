import urllib.request
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def fetch_xml_data():
   
    url = (
        'http://iszz.azo.hr/iskzl/rs/podatak/export/xml'
        '?postaja=160&polutant=5&tipPodatka=0'
        '&vrijemeOd=01.01.2017&vrijemeDo=31.12.2017'
    )
    response = urllib.request.urlopen(url)
    return ET.fromstring(response.read())


def parse_to_dataframe(root):
  
    records = []
    for item in root:
        try:
            measurement = float(item.find('vrijednost').text)
            timestamp = item.find('vrijeme').text
            records.append({'mjerenje': measurement, 'vrijeme': timestamp})
        except (AttributeError, ValueError, TypeError):
            continue

    df = pd.DataFrame(records)
    df['vrijeme'] = pd.to_datetime(df['vrijeme'], utc=True)
    df['month'] = df['vrijeme'].dt.month
    df['day_of_week'] = df['vrijeme'].dt.dayofweek
    return df


def plot_measurements(df):
 
    plt.figure(figsize=(12, 6))
    plt.plot(df['vrijeme'], df['mjerenje'], color='steelblue', linewidth=1)
    plt.title('PM10 koncentracija kroz vrijeme (2017, Osijek)')
    plt.xlabel('Datum')
    plt.ylabel('PM10 (µg/m³)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def top_concentration_days(df, top_n=3):
   
    return df.sort_values(by='mjerenje', ascending=False).head(top_n)


if __name__ == "__main__":
    xml_root = fetch_xml_data()
    df_pm10 = parse_to_dataframe(xml_root)
    plot_measurements(df_pm10)

    print("Top 3 dana s najvećom koncentracijom PM10:")
    print(top_concentration_days(df_pm10))
