use BOL;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- General
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Division into general classification 
select generalMatchClassification,
       count(*) [count],
       round(count(*) * 1e2 / (select count(*)from dbo.data_cleaned), 2) [percentage]
from dbo.data_cleaned
group by generalMatchClassification
order by 2 desc;

-- Division into detailed classification
select detailedMatchClassification,
       count(*) [count],
	   round(count(*) * 1e2 / (select count(*)from dbo.data_cleaned), 2) [percentage]
from dbo.data_cleaned
group by detailedMatchClassification
order by 2 desc;

-- Get an idea of all possible classifications
select noCancellation,
       onTimeDelivery,
       noCase,
       noReturn,
       detailedMatchClassification,
       count(*) [count],
	   round(count(*) * 1e2 / (select count(*)from dbo.data_cleaned), 2) [percentage]
from dbo.data_cleaned
where 1 = 1
      --and noCancellation = 1
      --and noReturn = 1
      --and noCase = 1
      --and onTimeDelivery = 1
group by noCancellation,
         onTimeDelivery,
         noCase,
         noReturn,
         detailedMatchClassification
order by 1,
		 2,
		 3,
		 4;

-- Price trend
select orderDate,
       sum(totalPrice)
from dbo.data_cleaned
group by orderDate
order by orderDate;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Link between unknown and briefpost
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
select transporterName,
       count(*) as [count],
       round(sum(case when generalMatchClassification like 'KNOWN HAPPY' then 1 else 0 end) * 1e2 / count(*), 2) percentage_happy,
       round(sum(case when generalMatchClassification like 'UNHAPPY' then 1 else 0 end) * 1e2 / count(*), 2) percentage_unhappy,
       round(sum(case when generalMatchClassification like 'UNKNOWN' then 1 else 0 end) * 1e2 / count(*), 2) percentage_unknown
from dbo.data_cleaned
group by transporterName
order by 3,2;

select round(sum(case when x.transporterName like '%brief%' then [count] else 0 end)*1e2 / sum(x.[count]), 2) as [briefpost],
       round(sum(case when x.transporterName not like '%brief%' then [count] else 0 end)*1e2 / sum(x.[count]), 2) as [anders]
from
(
    select transporterName,
           count(*) [count]
    from dbo.data_cleaned
    where generalMatchClassification like 'UNKNOWN'
    group by transporterName
) as x;

select generalMatchClassification,
       round(   count(*) * 1e2 /
                (
                    select count(*)
                    from dbo.data_cleaned
                    where transporterName like '%brief%'
                ),
                2
            ) as [percentage]
from dbo.data_cleaned
where transporterName like '%brief%'
group by generalMatchClassification;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Reason behind Unknown Matches
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
select sum(case when promisedDeliveryDate is null then 1 else 0 end) as [count unknown promised delivery date], -- occurs	  395 times ( 0.06%)
       sum(case when datetTimeFirstDeliveryMoment is null then 1 else 0 end) as [count unknown delivery moment] -- occurs 655,035 times (99.94%)
from dbo.data_2019
where generalMatchClassification like 'UNKNOWN';

select sum(case when promisedDeliveryDate is null then 1 else 0 end) as [count unknown promised delivery date], -- occurs	  469 times ( 0.05%)
       sum(case when datetTimeFirstDeliveryMoment is null then 1 else 0 end) as [count unknown delivery moment] -- occurs 860,461 times (99.95%)
from dbo.data_2020
where generalMatchClassification like 'UNKNOWN';

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Link between product groups and match classifications
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
select productGroup,
       count(*) as [count],
       round(sum(case when generalMatchClassification like 'KNOWN HAPPY' then 1 else 0 end) * 1e2 / count(*), 2) percentage_happy,
       round(sum(case when generalMatchClassification like 'UNHAPPY' then 1 else 0 end) * 1e2 / count(*), 2) percentage_unhappy,
       round(sum(case when generalMatchClassification like 'UNKNOWN' then 1 else 0 end) * 1e2 / count(*), 2) percentage_unknown
from dbo.data_cleaned
group by productGroup
order by 3;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Link between transporters and match classifications
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
select transporterName,
	   count(*) [count],
       round(sum(case when generalMatchClassification like 'KNOWN HAPPY' then 1 else 0 end)*1e2 / count(*), 2) percentage_happy,
       round(sum(case when generalMatchClassification like 'UNHAPPY' then 1 else 0 end)*1e2 / count(*), 2) percentage_unhappy,
       round(sum(case when generalMatchClassification like 'UNKNOWN' then 1 else 0 end)*1e2 / count(*), 2) percentage_unknown
from dbo.data_cleaned
group by transporterName
order by 3,2;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Link between product subgroups and match classifications
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
select productGroup,
       productSubGroup,
       count(*) as [count],
       round(sum(case when generalMatchClassification like 'KNOWN HAPPY' then 1 else 0 end) * 1e2 / count(*), 2) percentage_happy,
       round(sum(case when generalMatchClassification like 'UNHAPPY' then 1 else 0 end) * 1e2 / count(*), 2) percentage_unhappy,
       round(sum(case when generalMatchClassification like 'UNKNOWN' then 1 else 0 end) * 1e2 / count(*), 2) percentage_unknown
from dbo.data_cleaned
group by productGroup,
         productSubGroup
order by 1,
         4;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Link between sellers and match classifications
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
select sellerId,
       count(*) as [count],
       round(sum(case when generalMatchClassification like 'KNOWN HAPPY' then 1 else 0 end) * 1e2 / count(*), 2) percentage_happy,
       round(sum(case when generalMatchClassification like 'UNHAPPY' then 1 else 0 end) * 1e2 / count(*), 2) percentage_unhappy,
       round(sum(case when generalMatchClassification like 'UNKNOWN' then 1 else 0 end) * 1e2 / count(*), 2) percentage_unknown
from dbo.data_cleaned
group by sellerId
--having count(*) > 1000
order by 3 desc,
         2;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Link between order quantity and match classifications (not really a trend)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
select quantityOrdered,
       count(*) as [count],
       round(sum(case when generalMatchClassification like 'KNOWN HAPPY' then 1 else 0 end) * 1e2 / count(*), 2) percentage_happy,
       round(sum(case when generalMatchClassification like 'UNHAPPY' then 1 else 0 end) * 1e2 / count(*), 2) percentage_unhappy,
       round(sum(case when generalMatchClassification like 'UNKNOWN' then 1 else 0 end) * 1e2 / count(*), 2) percentage_unknown
from dbo.data_cleaned
group by quantityOrdered
order by 1;

select quantityOrdered,
	   count(*) as [count],
       sum(case when noReturn = 1 then 0 else 1 end) as [returns],
	   round(sum(case when noReturn = 1 then 0 else 1 end)*1e2 / count(*), 2) as percentage
from dbo.data_cleaned
group by quantityOrdered
order by 1

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Reasons for cancellations and returns
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
select cancellationReasonCode,
       count(*) [count],
       round(count(*) * 1e2 / (select count(*)from dbo.data_cleaned where cancellationDate is not null), 2) [percentage of cancellations]
from dbo.data_cleaned
where cancellationDate is not null
group by cancellationReasonCode
order by 2 desc;

select returnCode,
       count(*) [count],
       round(count(*) * 1e2 / (select count(*)from dbo.data_cleaned where returnDate is not null), 2) [percentage of returns]
from dbo.data_cleaned
where returnDate is not null
group by returnCode
order by 2 desc;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Co-occurrences of match determinants
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
drop table if exists #determinant_dependencies
select noCancellation,
       onTimeDelivery,
       noReturn,
       noCase,
       count(*) [count],
       round(count(*) * 1e2 / sum(count(*)) over (partition by noCancellation), 4) [percentage of noCancellation group],
       round(count(*) * 1e2 / sum(count(*)) over (partition by onTimeDelivery), 4) [percentage of onTimeDelivery group],
       round(count(*) * 1e2 / sum(count(*)) over (partition by noReturn), 4) [percentage of noReturn group],
       round(count(*) * 1e2 / sum(count(*)) over (partition by noCase), 4) [percentage of noCase group]
into #determinant_dependencies
from dbo.data_cleaned
group by noCancellation,
         onTimeDelivery,
         noReturn,
         noCase
order by 1,2,3,4;

select sum(x.[01. Percentage of cancellations with case])				as [01. Percentage of cancellations with case],					--  8.57%
       sum(x.[02. Percentage of cancellations with on time delivery])	as [02. Percentage of cancellations with on time delivery],		--  0.19%
       sum(x.[03. Percentage of cases with cancellation])				as [03. Percentage of cases with cancellation],					--  1.24%
       sum(x.[04. Percentage of cases with late delivery])				as [04. Percentage of cases with late delivery],				--  6.38%
       sum(x.[05. Percentage of cases with unknown delivery])			as [05. Percentage of cases with unknown delivery],				-- 52.96%
       sum(x.[06. Percentage of cases with return])						as [06. Percentage of cases with return],						-- 26.53%
       sum(x.[07. Percentage of late deliveries with case])				as [07. Percentage of late deliveries with case],				--  6.33%
       sum(x.[08. Percentage of late deliveries with return])			as [08. Percentage of late deliveries with return],				--  6.38%
       sum(x.[09. Percentage of unknown deliveries with cancellation])	as [09. Percentage of unknown deliveries with cancellation],	--  1.38%
       sum(x.[10. Percentage of unknown deliveries with case])			as [10. Percentage of unknown deliveries with case],			--  5.09%
       sum(x.[11. Percentage of unknown deliveries with return])		as [11. Percentage of unknown deliveries with return],			--  5.86%
       sum(x.[12. Percentage of returns with case])						as [12. Percentage of returns with case],						-- 15.54%
       sum(x.[13. Percentage of returns with late delivery])			as [13. Percentage of returns with late delivery],				--  3.76%
       sum(x.[14. Percentage of returns with unknown delivery])			as [14. Percentage of returns with unknown delivery]			-- 35.73%
from
(
    select case when noCancellation = 0 and noCase = 0				then sum([percentage of noCancellation group])	else 0 end [01. Percentage of cancellations with case],
           case when noCancellation = 0 and onTimeDelivery = 1		then sum([percentage of noCancellation group])	else 0 end [02. Percentage of cancellations with on time delivery],
           case when noCase = 0 and noCancellation = 0				then sum([percentage of noCase group])			else 0 end [03. Percentage of cases with cancellation],
           case when noCase = 0 and onTimeDelivery = 0				then sum([percentage of noCase group])			else 0 end [04. Percentage of cases with late delivery],
           case when noCase = 0 and onTimeDelivery is null			then sum([percentage of noCase group])			else 0 end [05. Percentage of cases with unknown delivery],
           case when noCase = 0 and noReturn = 0					then sum([percentage of noCase group])			else 0 end [06. Percentage of cases with return],
           case when onTimeDelivery = 0 and noCase = 0				then sum([percentage of onTimeDelivery group])	else 0 end [07. Percentage of late deliveries with case],
           case when onTimeDelivery = 0 and noReturn = 0			then sum([percentage of onTimeDelivery group])	else 0 end [08. Percentage of late deliveries with return],
           case when onTimeDelivery is null and noCancellation = 0	then sum([percentage of onTimeDelivery group])	else 0 end [09. Percentage of unknown deliveries with cancellation],
           case when onTimeDelivery is null and noCase = 0			then sum([percentage of onTimeDelivery group])	else 0 end [10. Percentage of unknown deliveries with case],
           case when onTimeDelivery is null and noReturn = 0		then sum([percentage of onTimeDelivery group])	else 0 end [11. Percentage of unknown deliveries with return],
           case when noReturn = 0 and noCase = 0					then sum([percentage of noReturn group])		else 0 end [12. Percentage of returns with case],
           case when noReturn = 0 and onTimeDelivery = 0			then sum([percentage of noReturn group])		else 0 end [13. Percentage of returns with late delivery],
           case when noReturn = 0 and onTimeDelivery is null		then sum([percentage of noReturn group])		else 0 end [14. Percentage of returns with unknown delivery]
    from #determinant_dependencies
    group by noCancellation,
             noCase,
             noReturn,
             onTimeDelivery
) as x;