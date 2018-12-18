import requests
from requests.exceptions import RequestException
import bs4
import csv

class Scraping:

    def __init__(self):
        self.ws = []

    def buscar_titulos(self, url_site = 'https://www.buscape.com.br/search/celular-e-smartphone',
                        numero_paginas = 2 ):
        #foram usado os urls: https://www.buscape.com.br/celular-e-smartphone
        #                     https://www.buscape.com.br/notebook
        #                     https://www.buscape.com.br/search/roupas
        #O primeiro url como fonte de exemplos positivos para o classificador e
        # os dois últimos como fontes de exemplos negativos.

        try:

            for i in range(1, numero_paginas + 1):
                payload = {'pagina':str(i)}
                response = requests.get(url_site, params = payload)
                soup = bs4.BeautifulSoup(response.text, features = 'html.parser')
                for div in soup.find_all(class_='card--product__name u-truncate-multiple-line'):
                       self.ws.append([
                       div.get_text('name')])

        except RequestException as e:
            print('erro:{}'.format(str(e)))
            
               
    def salvar_titulos(self, file_name = 'smartphone_buscape.csv'):

        with open(file_name, "w", newline = '', encoding = 'utf-8') as csvfile:
        	fieldname = ['produto']
        	writer = csv.DictWriter(csvfile, fieldnames=fieldname)
        	writer.writeheader()
        	for w in self.ws:
        		writer.writerow({'produto': w})

    def remover_lista(self):
        self.ws = []

if __name__ == "__main__":
    scraping = Scraping()
    scraping.buscar_titulos()
    scraping.salvar_titulos()
