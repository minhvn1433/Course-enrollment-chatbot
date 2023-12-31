import csv
import requests
from bs4 import BeautifulSoup

urls = [
    "https://catalog.mit.edu/subjects/1/",
    "https://catalog.mit.edu/subjects/2/",
    "https://catalog.mit.edu/subjects/3/",
    "https://catalog.mit.edu/subjects/4/",
    "https://catalog.mit.edu/subjects/5/",
    "https://catalog.mit.edu/subjects/6/",
    "https://catalog.mit.edu/subjects/7/",
    "https://catalog.mit.edu/subjects/8/",
    "https://catalog.mit.edu/subjects/9/",
    "https://catalog.mit.edu/subjects/10/",
    "https://catalog.mit.edu/subjects/11/",
    "https://catalog.mit.edu/subjects/12/",
    "https://catalog.mit.edu/subjects/14/",
    "https://catalog.mit.edu/subjects/15/",
    "https://catalog.mit.edu/subjects/16/",
    "https://catalog.mit.edu/subjects/17/",
    "https://catalog.mit.edu/subjects/18/",
    "https://catalog.mit.edu/subjects/20/",
    "https://catalog.mit.edu/subjects/21/",
    "https://catalog.mit.edu/subjects/21a/",
    "https://catalog.mit.edu/subjects/21h/",
    "https://catalog.mit.edu/subjects/21g/",
    "https://catalog.mit.edu/subjects/21l/",
    "https://catalog.mit.edu/subjects/21m/",
    "https://catalog.mit.edu/subjects/21w/",
    "https://catalog.mit.edu/subjects/22/",
    "https://catalog.mit.edu/subjects/24/",
    "https://catalog.mit.edu/subjects/as/",
    "https://catalog.mit.edu/subjects/cc/",
    "https://catalog.mit.edu/subjects/cms/",
    "https://catalog.mit.edu/subjects/csb/",
    "https://catalog.mit.edu/subjects/cse/",
    "https://catalog.mit.edu/subjects/ec/",
    "https://catalog.mit.edu/subjects/em/",
    "https://catalog.mit.edu/subjects/es/",
    "https://catalog.mit.edu/subjects/hst/",
    "https://catalog.mit.edu/subjects/ids/",
    "https://catalog.mit.edu/subjects/mas/",
    "https://catalog.mit.edu/subjects/ms/",
    "https://catalog.mit.edu/subjects/ns/",
    "https://catalog.mit.edu/subjects/scm/",
    "https://catalog.mit.edu/subjects/sp/",
    "https://catalog.mit.edu/subjects/sts/",
    "https://catalog.mit.edu/subjects/wgs/",
]

csv_file_path = "courses.csv"

fieldnames = [
    "Title",
    "Code",
    "Cluster",
    "Prerequisites",
    "Terms",
    "Hours",
    "Optional",
    "Description",
    "Instructors",
]

with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        course_blocks = soup.find_all("div", class_="courseblock")

        for block in course_blocks:
            title_tag = block.find("strong")
            cluster_tag = block.find("span", class_="courseblockcluster")
            prereq_tag = block.find("span", class_="courseblockprereq")
            terms_tag = block.find("span", class_="courseblockterms")
            hours_tag = block.find("span", class_="courseblockhours")
            optional_tag = block.find("span", class_="courseblockoptional")
            desc_tag = block.find("p", class_="courseblockdesc")
            instructors_tag = block.find("p", class_="courseblockinstructors seemore")

            title = " ".join(title_tag.text.split()[1:]) if title_tag else None
            code = title_tag.text.split()[0] if title_tag else None
            cluster = cluster_tag.text.replace("\n", "; ") if cluster_tag else "None"
            prereq = " ".join(prereq_tag.text.split()[1:]) if prereq_tag else "None"
            terms = terms_tag.text.replace("\n", "; ") if terms_tag else None
            hours = hours_tag.text if hours_tag else None
            optional = optional_tag.text if optional_tag else "None"
            desc = desc_tag.text if desc_tag else None
            instructors = instructors_tag.text if instructors_tag else None

            writer.writerow(
                {
                    "Title": title,
                    "Code": code,
                    "Cluster": cluster,
                    "Prerequisites": prereq,
                    "Terms": terms,
                    "Hours": hours,
                    "Optional": optional,
                    "Description": desc,
                    "Instructors": instructors,
                }
            )
