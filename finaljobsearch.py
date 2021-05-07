import requests
from bs4 import BeautifulSoup
import argparse
import pandas as pd

def scrape_jobs(URL):
    page = requests.get(URL)

    #print ("page: {}".format(page))
    soup = BeautifulSoup(page.content, "html.parser")
    #print ("soup: {}".format(soup))
    #results = soup.find(id="ResultsContainer")
    # Chandra
    results = soup.find(id="searchresults")
    #print (results)
    #pageDetails = soup.find("div", class_="pagination-label-row")
    pageDetails = soup.find("div", class_="well well-lg pagination-well pagination")
    #print (pageDetails)
    return results, pageDetails

def print_all_jobs(results):
    """Print details of all jobs returned by the search.
    The printed details are title, link, company name and location of the job.
    :param results: Parsed HTML container with all job listings
    :type results: BeautifulSoup object
    :return: None - just meant to print results
    """
    #job_elems = results.find_all("section", class_="card-content")
    # Chandra
    job_elems = results.find_all("tr", class_="data-row clickable")

    #print (job_elems)
    csvResults = []
    for job_elem in job_elems:
        # keep in mind that each job_elem is another BeautifulSoup object!
        # Chandra
        title_elem = job_elem.find("a", class_="jobTitle-link")
        #print (title_elem)
        #print(title_elem.text.strip())
        #print(title_elem["href"])
        joblink = "https://jobs.sap.com/"+title_elem["href"]

        page = requests.get(joblink)
        #print ("page: {}".format(page))
        soup = BeautifulSoup(page.content, "html.parser")
        #print ("soup: {}".format(soup))
        #results = soup.find(id="ResultsContainer")
        # Chandra
        #results = soup.find(id="similar-jobs")
        results = soup.find(id="innershell")
        jobDisplay = results.find("div", class_="jobDisplay")
        #print (jobDisplay)

        #jobTitle = jobDisplay.find("div", class_="jobTitle")
        jobTitle = jobDisplay.find(id="job-title")
        #if jobTitle == None:
        #    jobTitle = ""

        jobDate = jobDisplay.find("p", class_="jobDate")
        #if jobDate == None:
        #    jobDate = ""

        jobLocation = jobDisplay.find("span", class_="jobGeoLocation")
        #if jobLocation == None:
        #    jobLocation = ""

        if None in (jobTitle, jobDate, jobLocation):
            continue
        #print("job title: {}".format(jobTitle.text.strip()))
        #print ("job date: {}".format(jobDate.text.strip()))
        #print (jobLocation.text.strip())
        #print (jobLocation.text)

        jobDetails = jobDisplay.find("div", class_="job")
        #print (jobDetails.text)
            # print(job_elem.prettify())  # to inspect the 'None' element

        csvElem = [jobTitle.text.strip(), jobDate.text.strip(), jobLocation.text.strip(), jobDetails.text.strip()]

        csvResults.append(csvElem)
    
    dataF = pd.DataFrame(csvResults)
    dataF.columns = ["Title", "Date", "Location", "Details"]

    dataF.to_csv("jobResults.csv", header=True)

def store_job_details():
    jobList = pd.read_csv("jobResults.csv", index_col=0, header=0)

    jobDetails = jobList.loc[:, "Details"]

    jheaderList = []
    for index in range(len(jobDetails)):
        cindex = str(jobDetails[index]).find('COMPANY DESCRIPTION')
        #print (cindex)
        jheader = str(jobDetails[index])[0:cindex]
        if cindex == -1:
            jheader = ""
        #print (jheader)

        jheaderList.append(jheader)

    headerDictList = []
    for strList in jheaderList:
        newList = strList.split('\n')
        #print (newList)
        eleDict = {}
        for ele in newList:
            elelist = ele.split(': ')
            #print (elelist)
            if len(elelist) != 2:
                continue

            eleDict[elelist[0]] = elelist[1]

        headerDictList.append(eleDict)

    #print (headerDictList)

    listcolumns = ["Requisition ID", "Work Area", "Location", "Expected Travel", "Career Status", "Employment Type"]

    headerTable = []
    count = 0
    for dictele in headerDictList:
        headerTable.append([])
        if "Requisition ID" in dictele:
            headerTable[count].append(dictele["Requisition ID"])
        else:
            headerTable[count].append("")

        if "Work Area" in dictele:
            headerTable[count].append(dictele["Work Area"])
        else:
            headerTable[count].append("")
        if "Location" in dictele:
            headerTable[count].append(dictele["Location"])
        else:
            headerTable[count].append("")
        if "Expected Travel" in dictele:
            headerTable[count].append(dictele["Expected Travel"])
        else:
            headerTable[count].append("")
        if "Career Status" in dictele:
            headerTable[count].append(dictele["Career Status"])
        else:
            headerTable[count].append("")
        if "Employment Type" in dictele:
            headerTable[count].append(dictele["Employment Type"])
        else:
            headerTable[count].append("")

        count = count+1

    #print (headerTable)

    dataFheader = pd.DataFrame(headerTable)
    columnslist = ["Requisition ID", "Work Area", "Location", "Expected Travel", "Career Status", "Employment Type"]
    dataFheader.columns = columnslist
    #dataF.to_csv("jobResultsnew.csv", header=True)

    #jobListold = pd.read_csv("jobResults.csv", index_col=0, header=0)
    #jobListnew = pd.read_csv("jobResultsnew.csv", index_col=0, header=0)


    datajobListHeader = pd.concat([jobList, dataFheader], axis=1)
    #dataFnew.to_csv("jobResultsLatest.csv", header=True)

    # jobDetails initialized from the jobResults.csv above
    jsegmentList = []
    for index in range(len(jobDetails)):
        cindex = str(jobDetails[index]).find('Job Segment')
        #print (cindex)
        jsegment = str(jobDetails[index])[cindex:]
        if cindex == -1:
            jheader = ""
        #print (jsegment)

        jsegmentList.append(jsegment)

    jsegmentsTable = []
    for strele in jsegmentList:
        newList = strele.split(': \n')
        #print (newList)

        if len(newList) != 2:
            newList = ""
            jsegmentsTable.append("")
        else:
            jsegmentsTable.append(newList[1])

    #print (jsegmentsTable)

    dataFsegment = pd.DataFrame(jsegmentsTable)
    dataFsegment.columns = ["Job Segment"]
    #dataF.to_csv("jobResultsJobSegment.csv", header=True)

    #jobListold = pd.read_csv("jobResultsLatest.csv", index_col=0, header=0)
    #jobListnew = pd.read_csv("jobResultsJobSegment.csv", index_col=0, header=0)


    #dataFnew = pd.concat([jobListold, jobListnew], axis=1)
    dataFrameAll = pd.concat([datajobListHeader, dataFsegment], axis=1)
    #dataFrameAll.to_csv("jobResultsWithjobSegment.csv", header=True)

    return dataFrameAll

if __name__=="__main__":
    URL = "https://jobs.sap.com/search/?createNewAlert=false&q=&locationsearch=&optionsFacetsDD_department=&optionsFacetsDD_customfield3=&optionsFacetsDD_country="

    print ("Processing, please wait...")
    results, pageDetails = scrape_jobs(URL)
    print_all_jobs(results)

    page_elems = pageDetails.find("span", class_="srHelp")
    #print (page_elems.text)
    pagenums = page_elems.text.strip()
    pagenums = pagenums.split(" ")
    #print (pagenums)
    startpage = int(pagenums[1])
    endpage = int(pagenums[3])
    print ("start page: {}, end page: {}".format(startpage, endpage))

    dataFrameAllprev = store_job_details()
    for pagenum in range(endpage-2):

        print ("Page: {}".format(pagenum+1))
        title_text = "Page "+str(pagenum+2)
        nextpage = pageDetails.find("a", title=title_text)
        #print (nextpage)
        #print(nextpage.text.strip())
        #print(nextpage["href"])
        nextpagelink = "https://jobs.sap.com/search/"+nextpage["href"]

        results, pageDetails = scrape_jobs(nextpagelink)
        print_all_jobs(results)

        dataFrameAllcurrent = store_job_details()

        dataFrameTotal = pd.concat([dataFrameAllprev, dataFrameAllcurrent], axis=0)

        dataFrameAllprev = dataFrameTotal

        #if pagenum == 6:
        #    break

dataFrameTotal.to_csv("FinalJobDetails.csv", header=True, index=False)
