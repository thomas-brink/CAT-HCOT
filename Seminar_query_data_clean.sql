use Seminar;

DROP TABLE bol_data;

SELECT *
INTO bol_data
FROM data_load_1;

INSERT INTO bol_data
SELECT * FROM data_load_2;

-- Remove duplicate rows from the table -> 1.507.270 rows
SELECT DISTINCT * 
INTO #temp_bol_data
FROM bol_data;

-- Count nr. of rows in total data (non-duplicates): 1.507.270 rows
SELECT *
FROM #temp_bol_data;

-- CREATE CLEANED TABLE
-- First drop existing version of the table
DROP TABLE cleaned_bol_data;

-- Then move all rows from the non-duplicates table into the cleaned table
SELECT *
INTO cleaned_bol_data -- cleaned table
FROM #temp_bol_data;

-- Delete all noise (non-sensible rows)
DELETE FROM cleaned_bol_data
WHERE ((startDateCase < orderDate) OR (cancellationDate < orderDate) OR (promisedDeliveryDate < orderDate)
		OR (shipmentDate < orderDate) OR (CONVERT(date,datetTimeFirstDeliveryMoment) < orderDate) OR (returnDateTime < orderDate)
		OR (orderDate < registrationDateSeller) OR (cancellationDate > CONVERT(date,datetTimeFirstDeliveryMoment)) 
		OR (cancellationDate > returnDateTime) OR (returnDateTime < (CONVERT(date,datetTimeFirstDeliveryMoment)) 
		AND datetTimeFirstDeliveryMoment IS NOT NULL AND returnDateTime IS NOT NULL) OR (shipmentDate > returnDateTime)
		OR (shipmentDate > CONVERT(date,datetTimeFirstDeliveryMoment)) OR (registrationDateSeller IS NULL)
		OR (cancellationDate > shipmentDate AND (cancellationReasonCode = 'CUST_FE' OR cancellationReasonCode = 'CUST_CS')));

-- Count the rows in the resulting, cleaned table
SELECT COUNT(*) FROM cleaned_bol_data;

-- CHECKS
-- Check: boolean to classification correct
SELECT onTimeDelivery, noReturn, noCancellation, noCase, detailedMatchClassification, count(*)
FROM #temp_bol_data
GROUP BY onTimeDelivery, noReturn, noCancellation, noCase, detailedMatchClassification
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
		CASE WHEN (CASE WHEN (datediff(day,promisedDeliveryDate,datetTimeFirstDeliveryMoment) <= 0
					AND datediff(day,promisedDeliveryDate,datetTimeFirstDeliveryMoment) < 13) THEN 'True'
			  WHEN (datediff(day,promisedDeliveryDate,datetTimeFirstDeliveryMoment) > 0 
					AND datediff(day,promisedDeliveryDate,datetTimeFirstDeliveryMoment) < 13) THEN 'False'
			  ELSE 'NULL' END) <> onTimeDelivery THEN 1 ELSE 0 END AS diffOnTimeDeliveryCheck
FROM cleaned_bol_data)
SELECT diffReturnCheck, diffCaseCheck, diffCancellationCheck, diffOnTimeDeliveryCheck, count(*)
FROM diff_return_check
GROUP BY diffReturnCheck, diffCaseCheck, diffCancellationCheck, diffOnTimeDeliveryCheck;

-- All entries with a cancellation reason for which the noCancellation boolean is 1 (20.721 rows)
SELECT *
FROM cleaned_bol_data
WHERE cancellationReasonCode LIKE 'CUST%' AND noCancellation = 1;

SELECT TOP 100 * FROM cleaned_bol_data;

-- Get an idea of all possible classifications
SELECT noCancellation, onTimeDelivery, noCase, noReturn, detailedMatchClassification, count(*)
FROM cleaned_bol_data
--WHERE noCancellation = 1 AND noReturn = 1 AND noCase = 1
GROUP BY noCancellation, onTimeDelivery, noCase, noReturn, detailedMatchClassification;

-- Code to create table to be used for multi-class classification; run altogether
SET DATEFIRST 1;
DECLARE @looking_forward_days INTEGER = 5;
WITH transporter_classification AS (
SELECT transporterCode, count(*) AS nr_occurrences
FROM cleaned_bol_data
GROUP BY transporterCode )
SELECT	TOP 100 
		cbd.*, 
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
		END AS multi_class,
		@looking_forward_days as looking_forward_days, -- adjust yourself 
		CASE	WHEN (DATEDIFF(day,orderDate,CONVERT(DATE,datetTimeFirstDeliveryMoment)) < @looking_forward_days 
				  AND DATEDIFF(day,promisedDeliveryDate,CONVERT(DATE,datetTimeFirstDeliveryMoment)) <= 0)
					THEN 'On time'
				WHEN (DATEDIFF(day,orderDate,CONVERT(DATE,datetTimeFirstDeliveryMoment)) < @looking_forward_days
				  AND DATEDIFF(day,promisedDeliveryDate,CONVERT(DATE,datetTimeFirstDeliveryMoment)) > 0)
					THEN 'Late'
				ELSE 'Unknown'
		END AS delivery_category,
		CASE	WHEN (DATEDIFF(day,orderDate,CONVERT(DATE,startDateCase)) < @looking_forward_days)
					THEN 'Case started'
				ELSE 'Case not (yet) started'
		END AS case_category,
		CASE	WHEN (DATEDIFF(day,orderDate,returnDateTime) < @looking_forward_days)
					THEN 'Delivered, returned'
				WHEN (DATEDIFF(day,orderDate,CONVERT(DATE,datetTimeFirstDeliveryMoment)) < @looking_forward_days)
					THEN 'Delivered, not (yet) returned'
				ELSE 'Not yet delivered'
		END AS return_category,
		DATEPART(year,orderDate) AS order_year,
		FORMAT(DATEPART(month,orderDate),'00') AS order_month,
		CONCAT(DATEPART(year,orderDate),'-',FORMAT(DATEPART(month,orderDate),'00')) AS order_year_month,
		DATEPART(weekday,orderDate) as order_weekday,
		CASE	WHEN DATEPART(weekday,orderDate) <= 5
					THEN 0
				ELSE 1
		END AS weekend,
		CASE	WHEN orderDate > '2020-03-20'
					THEN 'Post-corona'
				ELSE 'Pre-corona'
		END AS corona_period,
		-- ADD HOLIDAYS? DON'T THINK SO, NOT EVEN 2 FULL YEARS OF DATA -> NON-SENSICAL PREDICTIONS
		CASE	WHEN tc.nr_occurrences > 100 
					THEN tc.transporterCode
				ELSE 'Other'
		END AS transporter_feature,
		DATEDIFF(month,registrationDateSeller,orderDate) AS partner_selling_months
FROM cleaned_bol_data as cbd
INNER JOIN transporter_classification as tc
	ON (cbd.transporterCode = tc.transporterCode);

SELECT transporterName, count(*)
FROM cleaned_bol_data
GROUP BY transporterName;

SELECT orderDate, sum(totalPrice)
FROM cleaned_bol_data
GROUP BY orderDate
ORDER BY orderDate;