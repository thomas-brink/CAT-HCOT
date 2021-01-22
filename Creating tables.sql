use BOL;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- CREATE IMPORTING AND CLEANED TABLES
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Create table for imported data
--create table dbo.data_imported (
--    orderDate date null,
--    productId bigint null,
--    sellerId int null,
--    totalPrice float null,
--    quantityOrdered int null,
--    countryCode nvarchar(max) null,
--    cancellationDate date null,
--    cancellationReasonCode nvarchar(max) null,
--    promisedDeliveryDate date null,
--    shipmentDate date null,
--    transporterCode varchar(max) null,
--    transporterName varchar(max) null,
--    transporterNameOther nvarchar(max) null,
--    dateTimeFirstDeliveryMoment datetime2 null,
--    fulfilmentType nvarchar(max) null,
--    startDateCase date null,
--    cntDistinctCaseIds int null,
--    returnDate date null,
--    quantityReturned int null,
--    returnCode nvarchar(max) null,
--    productTitle nvarchar(max) null,
--    brickName nvarchar(max) null,
--    chunkName nvarchar(max) null,
--    productGroup nvarchar(max) null,
--    productSubGroup nvarchar(max) null,
--    productSubSubGroup nvarchar(max) null,
--    registrationDateSeller date null,
--    countryOriginSeller nvarchar(max) null,
--    currentCountryAvailabilitySeller nvarchar(max) null,
--    calculationDefinitive nvarchar(max) null,
--    noCancellation bit null,
--    onTimeDelivery bit null,
--    noCase bit null,
--    hasOneCase bit null,
--    hasMoreCases bit null,
--    noReturn bit null,
--    detailedMatchClassification nvarchar(max) null,
--    generalMatchClassification nvarchar(max) null
--);

-- Create table for cleaned (combined) data
--create table dbo.data_cleaned (
--    orderDate date null,
--    productId bigint null,
--    sellerId int null,
--    totalPrice float null,
--    quantityOrdered int null,
--    countryCode nvarchar(max) null,
--    cancellationDate date null,
--    cancellationReasonCode nvarchar(max) null,
--    promisedDeliveryDate date null,
--    shipmentDate date null,
--    transporterCode varchar(max) null,
--    transporterName varchar(max) null,
--    transporterNameOther nvarchar(max) null,
--    dateTimeFirstDeliveryMoment datetime2 null,
--    fulfilmentType nvarchar(max) null,
--    startDateCase date null,
--    cntDistinctCaseIds int null,
--    returnDate date null,
--    quantityReturned int null,
--    returnCode nvarchar(max) null,
--    productTitle nvarchar(max) null,
--    brickName nvarchar(max) null,
--    chunkName nvarchar(max) null,
--    productGroup nvarchar(max) null,
--    productSubGroup nvarchar(max) null,
--    productSubSubGroup nvarchar(max) null,
--    registrationDateSeller date null,
--    countryOriginSeller nvarchar(max) null,
--    currentCountryAvailabilitySeller nvarchar(max) null,
--    calculationDefinitive nvarchar(max) null,
--    noCancellation bit null,
--    onTimeDelivery bit null,
--    noCase bit null,
--    hasOneCase bit null,
--    hasMoreCases bit null,
--    noReturn bit null,
--    detailedMatchClassification nvarchar(max) null,
--    generalMatchClassification nvarchar(max) null
--);

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- LOAD CREATED TABLES
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--delete dbo.data_imported
--where year(orderDate) = 2019;

--insert into dbo.data_imported
--select *
--from dbo.data_2019;
------------------------------------
--delete dbo.data_imported
--where year(orderDate) = 2020;

--insert into dbo.data_imported
--select *
--from dbo.data_2020;

----================================

--delete dbo.data_cleaned
--where year(orderDate) = 2019;

--insert into dbo.data_cleaned
--select *
--from dbo.data_2019;
------------------------------------
--delete dbo.data_cleaned
--where year(orderDate) = 2020;

--insert into dbo.data_cleaned
--select *
--from dbo.data_2020;
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Create table to be used for multi-class classification; run altogether
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
drop table if exists #transporter_classification;
select transporterCode,
       count(*) as nr_occurrences
into #transporter_classification
from dbo.data_cleaned
group by transporterCode;

drop table if exists dbo.data_combined_full;
set datefirst 1;
--declare @looking_forward_days integer = 5;
select cbd.*,
       case when (cbd.noCancellation = 1 and cbd.noReturn = 1 and cbd.noCase = 1 and cbd.onTimeDelivery = 'true') then
                'All good'
           when (cbd.noCancellation = 1 and cbd.noReturn = 1 and cbd.noCase = 1 and cbd.onTimeDelivery is null) then
               'Unknown delivery'
           when (cbd.noCancellation = 1 and cbd.noReturn = 1 and cbd.noCase = 1 and cbd.onTimeDelivery = 'false') then
               'Late delivery'
           when (cbd.noCancellation = 1 and cbd.noReturn = 1 and cbd.noCase = 0 and cbd.onTimeDelivery = 'true') then
               'Case'
           when (cbd.noCancellation = 1 and cbd.noReturn = 1 and cbd.noCase = 0 and cbd.onTimeDelivery is null) then
               'Case + Unknown delivery'
           when (cbd.noCancellation = 1 and cbd.noReturn = 1 and cbd.noCase = 0 and cbd.onTimeDelivery = 'false') then
               'Case + Late delivery'
           when (cbd.noCancellation = 1 and cbd.noReturn = 0 and cbd.noCase = 1 and cbd.onTimeDelivery = 'true') then
               'Return'
           when (cbd.noCancellation = 1 and cbd.noReturn = 0 and cbd.noCase = 1 and cbd.onTimeDelivery is null) then
               'Return + Unknown delivery'
           when (cbd.noCancellation = 1 and cbd.noReturn = 0 and cbd.noCase = 1 and cbd.onTimeDelivery = 'false') then
               'Return + Late delivery'
           when (cbd.noCancellation = 1 and cbd.noReturn = 0 and cbd.noCase = 0 and cbd.onTimeDelivery = 'true') then
               'Return + Case'
           when (cbd.noCancellation = 1 and cbd.noReturn = 0 and cbd.noCase = 0 and cbd.onTimeDelivery is null) then
               'Return + Case + Unknown delivery'
           when (cbd.noCancellation = 1 and cbd.noReturn = 0 and cbd.noCase = 0 and cbd.onTimeDelivery = 'false') then
               'Return + Case + Late delivery'
           when (cbd.noCancellation = 0 and cbd.noReturn = 1 and cbd.noCase = 0 and cbd.onTimeDelivery = 'true') then
               'Cancellation + Case'
           when (cbd.noCancellation = 0) then
               'Cancellation'
       end as determinantClassification,
       --@looking_forward_days as lookingForwardDays, -- adjust yourself 
       --case when
       --     (
       --         datediff(day, orderDate, convert(date, dateTimeFirstDeliveryMoment)) < @looking_forward_days
       --         and datediff(day, promisedDeliveryDate, convert(date, dateTimeFirstDeliveryMoment)) <= 0
       --     ) then
       --         'On time'
       --    when
       --    (
       --        datediff(day, orderDate, convert(date, dateTimeFirstDeliveryMoment)) < @looking_forward_days
       --        and datediff(day, promisedDeliveryDate, convert(date, dateTimeFirstDeliveryMoment)) > 0
       --    ) then
       --        'Late'
       --    else
       --        'Unknown'
       --end as deliveryCategory,
       --case when (datediff(day, orderDate, convert(date, startDateCase)) < @looking_forward_days) then
       --         'Case started'
       --    else
       --        'Case not (yet) started'
       --end as caseCategory,
       --case when (datediff(day, orderDate, returnDate) < @looking_forward_days) then
       --         'Delivered, returned'
       --    when (datediff(day, orderDate, convert(date, dateTimeFirstDeliveryMoment)) < @looking_forward_days) then
       --        'Delivered, not (yet) returned'
       --    else
       --        'Not yet delivered'
       --end as returnCategory,
       datepart(year, cbd.orderDate) as orderYear,
       format(datepart(month, cbd.orderDate), '00') as orderMonth,
       concat(datepart(year, cbd.orderDate), '-', format(datepart(month, cbd.orderDate), '00')) as orderYearMonth,
       datepart(weekday, cbd.orderDate) as orderWeekday,
       case when datepart(weekday, cbd.orderDate) <= 5 then 0 else 1 end as orderWeekend,
       case when cbd.orderDate > '2020-03-20' then 'Post-corona' else 'Pre-corona' end as orderCorona,
       case when tc.nr_occurrences > 100 then tc.transporterCode else 'Other' end as transporterFeature,
       len(cbd.productTitle) as productTitleLen,
       datediff(month, cbd.registrationDateSeller, cbd.orderDate) as partnerSellingMonths,
       datediff(day, cbd.orderDate, cbd.cancellationDate) as cancellationDays,
       datediff(day, cbd.orderDate, cbd.shipmentDate) as shipmentDays,
       datediff(day, cbd.orderDate, cbd.promisedDeliveryDate) as promisedDeliveryDays,
       datediff(day, cbd.orderDate, cbd.dateTimeFirstDeliveryMoment) as actualDeliveryDays,
       datediff(day, cbd.orderDate, cbd.startDateCase) as caseDays,
       datediff(day, cbd.orderDate, cbd.returnDate) as returnDays,
       case when cbd.countryCode = 'NL' then 1 else 0 end as countryCodeNL,
       case when cbd.fulfilmentType = 'FBB' then 1 else 0 end as fulfilmentByBol,
       case when cbd.countryOriginSeller = 'NL' then 1 else 0 end as countryOriginNL,
       case when cbd.countryOriginSeller = 'BE' then 1 else 0 end as countryOriginBE,
       case when cbd.countryOriginSeller = 'DE' then 1 else 0 end as countryOriginDE,
       case when datepart(weekday, cbd.orderDate) = 1 then 1 else 0 end as orderMonday,
       case when datepart(weekday, cbd.orderDate) = 2 then 1 else 0 end as orderTuesday,
       case when datepart(weekday, cbd.orderDate) = 3 then 1 else 0 end as orderWednesday,
       case when datepart(weekday, cbd.orderDate) = 4 then 1 else 0 end as orderThursday,
       case when datepart(weekday, cbd.orderDate) = 5 then 1 else 0 end as orderFriday,
       case when datepart(weekday, cbd.orderDate) = 6 then 1 else 0 end as orderSaturday,
       case when datepart(weekday, cbd.orderDate) = 7 then 1 else 0 end as orderSunday,
       case when format(datepart(month, cbd.orderDate), '00') = '01' then 1 else 0 end as orderJanuary,
       case when format(datepart(month, cbd.orderDate), '00') = '02' then 1 else 0 end as orderFebruary,
       case when format(datepart(month, cbd.orderDate), '00') = '03' then 1 else 0 end as orderMarch,
       case when format(datepart(month, cbd.orderDate), '00') = '04' then 1 else 0 end as orderApril,
       case when format(datepart(month, cbd.orderDate), '00') = '05' then 1 else 0 end as orderMay,
       case when format(datepart(month, cbd.orderDate), '00') = '06' then 1 else 0 end as orderJune,
       case when format(datepart(month, cbd.orderDate), '00') = '07' then 1 else 0 end as orderJuly,
       case when format(datepart(month, cbd.orderDate), '00') = '08' then 1 else 0 end as orderAugust,
       case when format(datepart(month, cbd.orderDate), '00') = '09' then 1 else 0 end as orderSeptember,
       case when format(datepart(month, cbd.orderDate), '00') = '10' then 1 else 0 end as orderOctober,
       case when format(datepart(month, cbd.orderDate), '00') = '11' then 1 else 0 end as orderNovember,
       case when format(datepart(month, cbd.orderDate), '00') = '12' then 1 else 0 end as orderDecember
into dbo.data_cleaned_full
from dbo.data_cleaned as cbd
    left join #transporter_classification as tc
        on (cbd.transporterCode = tc.transporterCode);