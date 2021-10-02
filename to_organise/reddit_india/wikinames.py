import pywikibot
from pywikibot import pagegenerators
site = pywikibot.site()
cat = pywikibot.Category(site, 'Category:Indian people')
gen = pagegenerators.CategorizedPageGenerator(cat)
for page in gen:
    print(page.title)
