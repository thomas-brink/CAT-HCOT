use Seminar;

DROP TABLE bol_data;
DROP TABLE data_2019;
DROP TABLE data_2020;

SELECT *
INTO bol_data
FROM data_2019;

INSERT INTO bol_data
SELECT * FROM data_2020;

ALTER TABLE bol_data
ALTER COLUMN totalPrice FLOAT;

-- Original data set -> 4.779.466 rows
SELECT COUNT(*) FROM bol_data;
SELECT COUNT(*) FROM cleaned_bol_data;
SELECT DISTINCT * FROM cleaned_bol_data;

-- Remove duplicate rows from the table -> 4.454.130 rows
SELECT DISTINCT * 
INTO #temp_bol_data
FROM bol_data;

-- Count nr. of rows in total data (non-duplicates): 4.779.466 rows
SELECT COUNT(*)
FROM bol_data;

-- CREATE CLEANED TABLE
-- First drop existing version of the table
DROP TABLE cleaned_bol_data;

-- Then move all rows from the non-duplicates table into the cleaned table
SELECT *
INTO cleaned_bol_data -- cleaned table
FROM bol_data;

-- Delete all noise (non-sensible rows) -> delete 6.516 rows
DELETE FROM cleaned_bol_data
WHERE ((startDateCase < orderDate) OR (cancellationDate < orderDate) OR (promisedDeliveryDate < orderDate)
		OR (shipmentDate < orderDate) OR (CONVERT(date,dateTimeFirstDeliveryMoment) < orderDate) OR (returnDateTime < orderDate)
		OR (orderDate < registrationDateSeller) OR (cancellationDate > CONVERT(date,dateTimeFirstDeliveryMoment)) 
		OR (cancellationDate > returnDateTime) OR (returnDateTime < (CONVERT(date,dateTimeFirstDeliveryMoment)) 
		AND dateTimeFirstDeliveryMoment IS NOT NULL AND returnDateTime IS NOT NULL) OR (shipmentDate > returnDateTime)
		OR (shipmentDate > CONVERT(date,dateTimeFirstDeliveryMoment)) OR (registrationDateSeller IS NULL)
		OR (promisedDeliveryDate IS NULL) 
		OR (cancellationDate > shipmentDate AND (cancellationReasonCode = 'CUST_FE' OR cancellationReasonCode = 'CUST_CS')));

-- Count the rows in the resulting, cleaned table -> 4.772.950 rows
SELECT COUNT(*) FROM cleaned_bol_data;

-- CHECKS
-- Check: boolean to classification correct
SELECT	onTimeDelivery, 
		noReturn, 
		noCancellation, 
		noCase, 
		generalMatchClassification, 
		COUNT(*)
FROM #temp_bol_data
GROUP BY onTimeDelivery, noReturn, noCancellation, noCase, generalMatchClassification
ORDER BY noCancellation, noCase, noReturn, onTimeDelivery;

-- Check whether the boolean variables are constructed correctly according to the related variables (CANCELLATIONS INCORRECT)
WITH diff_return_check AS (
SELECT cleaned_bol_data.*,
		(CASE WHEN (returnDateTime IS NOT NULL AND DATEDIFF(day,orderDate,returnDateTime) <= 29) THEN 0
			ELSE 1 END) - noReturn AS diffReturnCheck,
		(CASE WHEN (startDateCase IS NOT NULL AND DATEDIFF(day,orderDate,startDateCase) <= 29) THEN 0
			ELSE 1 END) - noCase AS diffCaseCheck,
		(CASE WHEN (cancellationDate IS NOT NULL AND DATEDIFF(day,orderDate,cancellationDate) <= 10) THEN 0
			ELSE 1 END) - noCancellation AS diffCancellationCheck,
		CASE WHEN (CASE WHEN (DATEDIFF(day,promisedDeliveryDate,dateTimeFirstDeliveryMoment) <= 0
					AND DATEDIFF(day,promisedDeliveryDate,dateTimeFirstDeliveryMoment) < 13) THEN 'True'
			  WHEN (DATEDIFF(day,promisedDeliveryDate,dateTimeFirstDeliveryMoment) > 0 
					AND DATEDIFF(day,promisedDeliveryDate,dateTimeFirstDeliveryMoment) < 13) THEN 'False'
			  ELSE 'NULL' END) <> onTimeDelivery THEN 1 ELSE 0 END AS diffOnTimeDeliveryCheck
FROM cleaned_bol_data)
SELECT diffReturnCheck, diffCaseCheck, diffCancellationCheck, diffOnTimeDeliveryCheck, count(*)
FROM diff_return_check
GROUP BY diffReturnCheck, diffCaseCheck, diffCancellationCheck, diffOnTimeDeliveryCheck;

-- All entries with a cancellation reason for which the noCancellation boolean is 1 (51.579 rows)
SELECT *
FROM cleaned_bol_data
WHERE cancellationReasonCode IS NOT NULL 
	AND noCancellation = 1
	AND DATEDIFF(DAY,orderDate,cancellationDate) <= 10;

SELECT *
FROM cleaned_bol_data
WHERE noCancellation = 0;

SELECT TOP 100 * FROM cleaned_bol_data;

-- Get an idea of all possible classifications
SELECT	noCancellation, 
		onTimeDelivery, 
		noCase, 
		noReturn, 
		generalMatchClassification, 
		count(*)
FROM cleaned_bol_data
--WHERE noCancellation = 1 AND noReturn = 1 AND noCase = 1
GROUP BY noCancellation, onTimeDelivery, noCase, noReturn, generalMatchClassification;

-- Division into general classification 
SELECT	generalMatchClassification,
		COUNT(*)
FROM cleaned_bol_data
GROUP BY generalMatchClassification;

-- Division into detailed classification
SELECT	detailedMatchClassification,
		COUNT(*)
FROM cleaned_bol_data
GROUP BY detailedMatchClassification;

-- Code to create table to be used for multi-class classification; run altogether
DROP transporter_classification;
SELECT transporterCode, count(*) AS nr_occurrences
INTO transporter_classification
FROM cleaned_bol_data
GROUP BY transporterCode;

DROP TABLE cleaned_bol_data_full;
SET DATEFIRST 1;
DECLARE @looking_forward_days INTEGER = 5;
SELECT	cbd.*, 
		CASE	WHEN (noCancellation = 1 AND noReturn = 1 AND noCase = 1 AND onTimeDelivery = 'true')
					THEN 'All good'
				WHEN (noCancellation = 1 AND noReturn = 1 AND noCase = 1 AND onTimeDelivery IS NULL)
					THEN 'Unknown delivery'
				WHEN (noCancellation = 1 AND noReturn = 1 AND noCase = 1 AND onTimeDelivery = 'false')
					THEN 'Late delivery'
				WHEN (noCancellation = 1 AND noReturn = 1 AND noCase = 0 AND onTimeDelivery = 'true')
					THEN 'Case'
				WHEN (noCancellation = 1 AND noReturn = 1 AND noCase = 0 AND onTimeDelivery IS NULL)
					THEN 'Case + Unknown delivery'
				WHEN (noCancellation = 1 AND noReturn = 1 AND noCase = 0 AND onTimeDelivery = 'false')
					THEN 'Case + Late delivery'
				WHEN (noCancellation = 1 AND noReturn = 0 AND noCase = 1 AND onTimeDelivery = 'true')
					THEN 'Return'
				WHEN (noCancellation = 1 AND noReturn = 0 AND noCase = 1 AND onTimeDelivery IS NULL)
					THEN 'Return + Unknown delivery'
				WHEN (noCancellation = 1 AND noReturn = 0 AND noCase = 1 AND onTimeDelivery = 'false')
					THEN 'Return + Late delivery'
				WHEN (noCancellation = 1 AND noReturn = 0 AND noCase = 0 AND onTimeDelivery = 'true')
					THEN 'Return + Case'
				WHEN (noCancellation = 1 AND noReturn = 0 AND noCase = 0 AND onTimeDelivery IS NULL)
					THEN 'Return + Case + Unknown delivery'
				WHEN (noCancellation = 1 AND noReturn = 0 AND noCase = 0 AND onTimeDelivery = 'false')
					THEN 'Return + Case + Late delivery'
				WHEN (noCancellation = 0 AND noReturn = 1 AND noCase = 0 AND onTimeDelivery = 'true')
					THEN 'Cancellation + Case'
				WHEN (noCancellation = 0)
					THEN 'Cancellation'
		END AS determinantClassification,
		--@looking_forward_days as lookingForwardDays, -- adjust yourself 
		--CASE	WHEN (DATEDIFF(day,orderDate,CONVERT(DATE,dateTimeFirstDeliveryMoment)) < @looking_forward_days 
		--		  AND DATEDIFF(day,promisedDeliveryDate,CONVERT(DATE,dateTimeFirstDeliveryMoment)) <= 0)
		--			THEN 'On time'
		--		WHEN (DATEDIFF(day,orderDate,CONVERT(DATE,dateTimeFirstDeliveryMoment)) < @looking_forward_days
		--		  AND DATEDIFF(day,promisedDeliveryDate,CONVERT(DATE,dateTimeFirstDeliveryMoment)) > 0)
		--			THEN 'Late'
		--		ELSE 'Unknown'
		--END AS deliveryCategory,
		--CASE	WHEN (DATEDIFF(day,orderDate,CONVERT(DATE,startDateCase)) < @looking_forward_days)
		--			THEN 'Case started'
		--		ELSE 'Case not (yet) started'
		--END AS caseCategory,
		--CASE	WHEN (DATEDIFF(day,orderDate,returnDateTime) < @looking_forward_days)
		--			THEN 'Delivered, returned'
		--		WHEN (DATEDIFF(day,orderDate,CONVERT(DATE,dateTimeFirstDeliveryMoment)) < @looking_forward_days)
		--			THEN 'Delivered, not (yet) returned'
		--		ELSE 'Not yet delivered'
		--END AS returnCategory,
		DATEPART(year,orderDate) AS orderYear,
		FORMAT(DATEPART(month,orderDate),'00') AS orderMonth,
		CONCAT(DATEPART(year,orderDate),'-',FORMAT(DATEPART(month,orderDate),'00')) AS orderYearMonth,
		DATEPART(weekday,orderDate) AS orderWeekday,
		CASE	WHEN DATEPART(weekday,orderDate) <= 5
					THEN 0
				ELSE 1
		END AS orderWeekend,
		CASE	WHEN orderDate > '2020-03-20'
					THEN 'Post-corona'
				ELSE 'Pre-corona'
		END AS orderCorona,
		CASE	WHEN tc.nr_occurrences > 100 
					THEN tc.transporterCode
				ELSE 'Other'
		END AS transporterFeature,
		DATEDIFF(month,registrationDateSeller,orderDate) AS partnerSellingMonths,
		DATEDIFF(day,orderDate,cancellationDate) AS cancellationDays,
		DATEDIFF(day,orderDate,shipmentDate) AS shipmentDays,
		DATEDIFF(day,orderDate,promisedDeliveryDate) AS promisedDeliveryDays,
		DATEDIFF(day,orderDate,dateTimeFirstDeliveryMoment) AS actualDeliveryDays,
		DATEDIFF(day,orderDate,startDateCase) AS caseDays,
		DATEDIFF(day,orderDate,returnDateTime) AS returnDays,
		CASE WHEN countryCode = 'NL' THEN 1 ELSE 0 END AS countryCodeNL,
		CASE WHEN fulfilmentType = 'FBB' THEN 1 ELSE 0 END AS fulfilmentByBol,
		CASE WHEN countryOriginSeller = 'NL' THEN 1 ELSE 0 END AS countryOriginNL,
		CASE WHEN countryOriginSeller = 'BE' THEN 1 ELSE 0 END AS countryOriginBE,
		CASE WHEN countryOriginSeller = 'DE' THEN 1 ELSE 0 END AS countryOriginDE,
		CASE WHEN DATEPART(weekday,orderDate) = 1 THEN 1 ELSE 0 END AS orderMonday,
		CASE WHEN DATEPART(weekday,orderDate) = 2 THEN 1 ELSE 0 END AS orderTuesday,
		CASE WHEN DATEPART(weekday,orderDate) = 3 THEN 1 ELSE 0 END AS orderWednesday,
		CASE WHEN DATEPART(weekday,orderDate) = 4 THEN 1 ELSE 0 END AS orderThursday,
		CASE WHEN DATEPART(weekday,orderDate) = 5 THEN 1 ELSE 0 END AS orderFriday,
		CASE WHEN DATEPART(weekday,orderDate) = 6 THEN 1 ELSE 0 END AS orderSaturday,
		CASE WHEN DATEPART(weekday,orderDate) = 7 THEN 1 ELSE 0 END AS orderSunday,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '01' THEN 1 ELSE 0 END AS orderJanuary,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '02' THEN 1 ELSE 0 END AS orderFebruary,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '03' THEN 1 ELSE 0 END AS orderMarch,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '04' THEN 1 ELSE 0 END AS orderApril,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '05' THEN 1 ELSE 0 END AS orderMay,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '06' THEN 1 ELSE 0 END AS orderJune,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '07' THEN 1 ELSE 0 END AS orderJuly,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '08' THEN 1 ELSE 0 END AS orderAugust,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '09' THEN 1 ELSE 0 END AS orderSeptember,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '10' THEN 1 ELSE 0 END AS orderOctober,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '11' THEN 1 ELSE 0 END AS orderNovember,
		CASE WHEN FORMAT(DATEPART(month,orderDate),'00') = '12' THEN 1 ELSE 0 END AS orderDecember
INTO cleaned_bol_data_full
FROM cleaned_bol_data as cbd
LEFT JOIN transporter_classification as tc
	ON (cbd.transporterCode = tc.transporterCode);

ALTER TABLE cleaned_bol_data_full
ADD productTitleLength AS LEN(productTitle);

SELECT TOP 100 * FROM cleaned_bol_data_full;

-- Some checks
SELECT transporterName, COUNT(*)
FROM cleaned_bol_data
GROUP BY transporterName;

SELECT orderDate, SUM(totalPrice)
FROM cleaned_bol_data
GROUP BY orderDate
ORDER BY orderDate;

SELECT DISTINCT returnCode
FROM cleaned_bol_data;
SELECT TOP 100 * FROM cleaned_bol_data;

SELECT transporterCode, count(*)
FROM cleaned_bol_data
GROUP BY transporterCode;

-- Analysis on Unknown Matches for the Data section
SELECT	generalMatchClassification, 
		COUNT(*) 
FROM cleaned_bol_data 
WHERE promisedDeliveryDate IS NULL
GROUP BY generalMatchClassification;

SELECT	*
FROM cleaned_bol_data
WHERE dateTimeFirstDeliveryMoment IS NOT NULL 
		AND generalMatchClassification = 'UNKNOWN'
		AND DATEDIFF(DAY,orderDate,dateTimeFirstDeliveryMoment) < 13;

SELECT	SUM(CASE WHEN dateTimeFirstDeliveryMoment IS NULL THEN 1 ELSE 0 END) AS null_delivery_moment,
		COUNT(*)
FROM cleaned_bol_data WHERE generalMatchClassification = 'UNKNOWN';

SELECT *
FROM cleaned_bol_data
WHERE generalMatchClassification = 'UNKNOWN';

SELECT * FROM cleaned_bol_data
WHERE productId = '9200000078766106' AND orderDate = '2019-03-08';

SELECT * FROM cleaned_bol_data;

-- Check co-occurrences of booleans -> 8.6% of cancellations has case, 0.2% of cancellations delivered on time
-- 1.2% of cases has cancellation, 6.4% of cases has late delivery, 53.0% of cases has NULL delivery
-- 26.5% of cases has return, 6.3% of late deliveries has case, 6.4% of late deliveries has return
-- 1.4% of NULL deliveries has cancellation, 5.1% of NULL deliveries has case, 5.9% of NULL deliveries has return
-- 15.5% of returns has case, 3.8% of returns has late delivery, 35.7% of returns has NULL delivery
SELECT onTimeDelivery, COUNT(*)
FROM cleaned_bol_data
WHERE noReturn = 0
GROUP BY onTimeDelivery;

-- Decision tree validation -> Unknown delivery should be further down the tree
SELECT generalMatchClassification, noCancellation, COUNT(*)
FROM cleaned_bol_data
WHERE noReturn = 0 AND onTimeDelivery IS NULL
GROUP BY generalMatchClassification, noCancellation;

-- Remove rows with NULL promisedDeliveryDate?
SELECT *
FROM cleaned_bol_data
WHERE promisedDeliveryDate IS NULL;

SELECT currentCountryAvailabilitySeller, COUNT(*)
FROM cleaned_bol_data
GROUP BY currentCountryAvailabilitySeller;

SELECT sellerId, COUNT(DISTINCT currentCountryAvailabilitySeller)
FROM cleaned_bol_data
GROUP BY sellerId
ORDER BY COUNT(DISTINCT currentCountryAvailabilitySeller) DESC;

SELECT * FROM cleaned_bol_data
WHERE quantityReturned > quantityOrdered;

SELECT * FROM cleaned_bol_data
WHERE noReturn = 0;

-- Check number of occurrences per product
SELECT productId, COUNT(*)
FROM cleaned_bol_data_full
GROUP BY productId
ORDER BY COUNT(*) DESC;

-- Check number of occurrences per sellerId
SELECT sellerId, COUNT(*)
FROM cleaned_bol_data_full
GROUP BY sellerId
ORDER BY COUNT(*) DESC;

-- Check fraction of return quantity per productId
SELECT productId, SUM(ISNULL(quantityReturned,0)) as totalQuantityReturned, SUM(quantityOrdered) as totalQuantityOrdered
FROM cleaned_bol_data_full
GROUP BY productId;

-- Check fraction of return quantity per sellerId
SELECT sellerId, SUM(ISNULL(quantityReturned,0)) as totalQuantityReturned, SUM(quantityOrdered) as totalQuantityOrdered
FROM cleaned_bol_data_full
GROUP BY sellerId;
