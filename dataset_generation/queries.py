""" Queries for BugBox metadata retrieval """

from string import Template

images = Template(
    """SELECT LTRIM(image, '/') AS image, uuid, classification_id, morphospecie_id
    FROM (cms_app_specimen AS s 
        JOIN cms_app_image AS i 
        ON s.id = i.specimen_id) 
    WHERE s.acceptance = 2 AND s.classification_id > 0 AND s.sample_id IS NOT NULL AND i.date_added > NOW() - INTERVAL \'1 $lookback\'"""
)

images_with_taxon = Template(
    f"""SELECT image, uuid AS specimen_id, m.name AS morhospecies, classification_id AS taxon_id, m.id AS morphospecie_id, t.order, t.family, t.genus
    FROM ($image_query) AS images 
        JOIN taxon_app_taxon as t 
        ON classification_id = \"taxonID\"
        JOIN taxon_app_morphospecie as m
        ON images.morphospecie_id = m.id"""
)

taxa = \
    """SELECT "taxonID" AS taxon_id, CONCAT("order", ' ', family, ' ', genus ) AS class_name
    FROM taxon_app_taxon
    WHERE "taxonRank" = 'genus' AND genus = "canonicalName" AND "taxonomicStatus" = 'accepted'
        AND "order" <> '' AND "family" <> ''"""

reference_images = \
    """SELECT LTRIM(image, '/') AS image, t.order, t.family, t.genus
    FROM (refimages_app_referenceimage AS r 
        JOIN taxon_app_taxon AS t
        ON classification_id = \"taxonID\") 
    WHERE r.classification_id > 0"""

morphospecies = \
    """SELECT * FROM taxon_app_morphospecie WHERE taxon_id IS NOT NULL"""
