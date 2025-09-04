BEGIN TRANSACTION;

DROP TABLE IF EXISTS d260_weerstanden_skims_auto_inclusief_grensovergangen;
DROP TYPE IF EXISTS d260_tijd_type;

CREATE TYPE d260_tijd_type AS ENUM (
	'ochtendspits',
	'avondspits',
	'restdag'
);

END TRANSACTION;
