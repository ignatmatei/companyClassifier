def DFT (labels, description, business_tags, sector, category, niche) :
    # results is a list of dictionaries
    results = []
    for i in range(len(description)) :
        for label in labels :
            val = 0
            #check how many times the label is included in the description
            if label in description[i] :
                val += 0.1
            #check how many times the label is included in the business_tags
            for business_tag in business_tags[i] :
                if label in business_tag :
                    val += 0.1
            if sector[i] in label :
                val += 0.5
            if category[i] in label :
                val += 1
            if niche[i] in label :
                val += 1.5
            

    return results

