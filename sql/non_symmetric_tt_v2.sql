COPY (
  WITH pc4_mapped AS (
    SELECT 
      tt.tijd,
      tt.reistijd_in_seconden,
      tt.afstand_in_meters,
      pc4_from.pc4_code AS pc4_from,
      pc4_to.pc4_code AS pc4_to
    FROM d260_weerstanden_skims_auto_inclusief_grensovergangen tt
    JOIN df_centroids_with_pc4 pc4_from ON tt.from_zone_gid = pc4_from.gid
    JOIN df_centroids_with_pc4 pc4_to ON tt.to_zone_gid = pc4_to.gid
    WHERE pc4_from.pc4_code IS NOT NULL
      AND pc4_to.pc4_code IS NOT NULL
  ),

  avond_tt AS (
    SELECT 
      pc4_from, pc4_to,
      AVG(reistijd_in_seconden) AS avond_tt
    FROM pc4_mapped
    WHERE tijd = 'avondspits'
    GROUP BY pc4_from, pc4_to
  ),

  ochtend_tt AS (
    SELECT 
      pc4_from, pc4_to,
      AVG(reistijd_in_seconden) AS ochtend_tt
    FROM pc4_mapped
    WHERE tijd = 'ochtendspits'
    GROUP BY pc4_from, pc4_to
  ),

  normal_tt AS (
    SELECT 
      pc4_from, pc4_to,
      AVG(reistijd_in_seconden) AS normal_tt
    FROM pc4_mapped
    WHERE tijd = 'restdag'
    GROUP BY pc4_from, pc4_to
  ),

  afstand_tt AS (
    SELECT 
      pc4_from, pc4_to,
      AVG(afstand_in_meters) AS afstand
    FROM pc4_mapped
	WHERE tijd = 'restdag'
    GROUP BY pc4_from, pc4_to
  )

  SELECT 
    COALESCE(a.pc4_from, o.pc4_from, n.pc4_from, af.pc4_from) AS pc4_from,
    COALESCE(a.pc4_to, o.pc4_to, n.pc4_to, af.pc4_to) AS pc4_to,
    a.avond_tt,
    o.ochtend_tt,
    n.normal_tt,
    af.afstand
  FROM avond_tt a
  FULL OUTER JOIN ochtend_tt o
    ON a.pc4_from = o.pc4_from AND a.pc4_to = o.pc4_to
  FULL OUTER JOIN normal_tt n
    ON COALESCE(a.pc4_from, o.pc4_from) = n.pc4_from
    AND COALESCE(a.pc4_to, o.pc4_to) = n.pc4_to
  FULL OUTER JOIN afstand_tt af
    ON COALESCE(a.pc4_from, o.pc4_from, n.pc4_from) = af.pc4_from
    AND COALESCE(a.pc4_to, o.pc4_to, n.pc4_to) = af.pc4_to
)
TO 'C:/Project/Test/forYasi/all_nonsymmetric_edited_output_v9.csv' 
WITH (FORMAT csv, HEADER true);
