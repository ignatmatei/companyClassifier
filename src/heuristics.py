def DFT (labels, companies , threshold) :
    # results is a list of dictionaries
    results = {}
    for i , company in enumerate(companies) :
        for label in labels :
            val = 0
            #check how many times the label is included in the description
            if label in str(company['description']) :
                val += 0.1
            #check how many times the label is included in the business_tags
            for business_tag in company['bussines_tags'] :
                if label in str(business_tag) :
                    val += 0.1
            if str(company['sector']) in label :
                val += 0.5
            if str(company['category']) in label :
                val += 1
            if str(company['niche']) in label :
                val += 1.5
            if val > threshold :
                res =(label, val)
                results[i] = res

    return results

